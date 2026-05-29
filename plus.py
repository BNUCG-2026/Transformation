import taichi as ti
import numpy as np
import math

# Initialize Taichi to run on CPU backend
ti.init(arch=ti.cpu)

# 8 vertices of a standard cube (length = 2, centered at origin)
cube_vertices_np = [
    [-1.0, -1.0, -1.0],  # 0
    [ 1.0, -1.0, -1.0],  # 1
    [ 1.0,  1.0, -1.0],  # 2
    [-1.0,  1.0, -1.0],  # 3
    [-1.0, -1.0,  1.0],  # 4
    [ 1.0, -1.0,  1.0],  # 5
    [ 1.0,  1.0,  1.0],  # 6
    [-1.0,  1.0,  1.0]   # 7
]

# 12 edges of the cube connecting the vertices
cube_edges_np = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Back face
    [4, 5], [5, 6], [6, 7], [7, 4],  # Front face
    [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
]

# Taichi Fields
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
edges = ti.Vector.field(2, dtype=ti.i32, shape=12)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# Initialize fields with valid NumPy array layouts
vertices.from_numpy(np.array(cube_vertices_np, dtype=np.float32))
edges.from_numpy(np.array(cube_edges_np, dtype=np.int32))


@ti.func
def quaternion_to_matrix(q) -> ti.types.matrix(4, 4, ti.f32):
    """
    Convert a normalized quaternion [w, x, y, z] into a 4x4 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ti.Matrix([
        [1.0 - 2.0*(y*y + z*z),       2.0*(x*y - w*z),       2.0*(x*z + w*y), 0.0],
        [      2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z),       2.0*(y*z - w*x), 0.0],
        [      2.0*(x*z - w*y),       2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y), 0.0],
        [                  0.0,                   0.0,                   0.0, 1.0]
    ])


@ti.func
def quaternion_slerp(q0, q1, t: ti.f32):
    """
    Spherical Linear Interpolation (Slerp) between two quaternions.
    """
    dot = q0.dot(q1)
    
    q1_target = q1
    if dot < 0.0:
        dot = -dot
        q1_target = -q1
        
    omega = ti.acos(ti.max(ti.min(dot, 1.0), -1.0))
    sin_omega = ti.sin(omega)
    
    w0 = 1.0 - t
    w1 = t
    if sin_omega > 0.001:
        w0 = ti.sin((1.0 - t) * omega) / sin_omega
        w1 = ti.sin(t * omega) / sin_omega
        
    return w0 * q0 + w1 * q1_target


@ti.func
def get_view_matrix(eye_pos):
    """
    View matrix to reposition the camera layout.
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    Perspective Projection Matrix.
    """
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return M_ortho_scale @ M_ortho_trans @ M_p2o


@ti.kernel
def compute_transform(t: ti.f32):
    """
    Computes vertex transformations dynamically using Slerp rotation interpolation.
    """
    eye_pos = ti.Vector([0.0, 0.0, 6.0])
    
    # Define R0 state quaternion (Euler equivalent: pitch=-30, yaw=-30)
    q0 = ti.Vector([0.883, -0.306, -0.306, 0.177])
    q0 = q0.normalized()
    
    # Define R1 state quaternion (Euler equivalent: pitch=60, yaw=60)
    q1 = ti.Vector([0.500, 0.500, 0.500, 0.500])
    q1 = q1.normalized()
    
    # Spherical linear interpolation between R0 and R1 orientation
    qt = quaternion_slerp(q0, q1, t)
    model = quaternion_to_matrix(qt)
    
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # Perspective division to Normalized Device Coordinates (NDC)
        v_ndc = v_clip / v_clip[3]
        
        # Viewport Mapping to GUI canvas context [0, 1] x [0, 1]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0


def main():
    gui = ti.GUI("3D Cube Interpolation (Taichi)", res=(700, 700))
    
    # Interpolation factor t ranges from 0.0 to 1.0
    t = 0.0
    direction = 1.0
    speed = 0.01  # Animation transition speed
    
    print("==================================================")
    print("Controls:")
    print("  Press 'A' to decrease interpolation factor manually")
    print("  Press 'D' to increase interpolation factor manually")
    print("  Press 'SPACE' to toggle auto-ping-pong animation")
    print("==================================================")
    
    auto_animate = True
    
    while gui.running:
        # Handle user triggers safely
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                auto_animate = False
                t = max(0.0, t - 0.05)
            elif gui.event.key == 'd':
                auto_animate = False
                t = min(1.0, t + 0.05)
            elif gui.event.key == ' ':
                auto_animate = not auto_animate
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # Drive automatic ping-pong interpolation loop
        if auto_animate:
            t += direction * speed
            if t >= 1.0:
                t = 1.0
                direction = -1.0
            elif t <= 0.0:
                t = 0.0
                direction = 1.0
                
        # Perform parallel transformation operations via Taichi Parallel Architecture
        compute_transform(t)
        
        # Clear frame background implicitly on gui.show(), draw wireframe paths
        for idx in range(12):
            edge = cube_edges_np[idx]
            p1 = screen_coords[edge[0]]
            p2 = screen_coords[edge[1]]
            
            # Draw wireframe lines using an attractive gradient hue matching interpolation status
            gui.line(p1, p2, radius=2, color=ti.rgb_to_hex((1.0 - t, 0.5 + t * 0.5, t)))
            
        gui.show()


if __name__ == '__main__':
    main()
