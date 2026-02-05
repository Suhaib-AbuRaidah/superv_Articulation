import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
class VisualizerWrapper:
    def __init__(env):
        env.vis = o3d.visualization.Visualizer()
        
        env.added = False

        # Placeholders for geometries


    def initialize(env, pc_current, pc_target,corresp_transformed=None, corresp_target=None):
        # Set initial points and colors
        env.pc1 = o3d.geometry.PointCloud()
        env.pc2 = o3d.geometry.PointCloud()
        env.vis.create_window(window_name="RL Alignment Viewer", width=800, height=600)
        env.pc1.points = o3d.utility.Vector3dVector(np.asarray(pc_current))
        env.pc1.paint_uniform_color([0.5, 0.5, 0.5])  # Red
        if corresp_transformed is not None and corresp_target is not None:
            env.line_set = o3d.geometry.LineSet()
            lines = [[i, i + len(corresp_transformed)] for i in range(len(corresp_transformed))]
            np.random.seed(0)
            line_colors = [np.random.rand(3) for _ in range(len(lines))]
            
            all_points = np.vstack((corresp_transformed, corresp_target))
            env.line_set = o3d.utility.Vector3dVector(all_points)
            env.line_set.lines = o3d.utility.Vector2iVector(lines)
            env.line_set.colors = o3d.utility.Vector3dVector(line_colors)
        env.axis_steps = 40
        env.pc2.points = o3d.utility.Vector3dVector(np.asarray(pc_target))
        env.pc2.paint_uniform_color([0.5, 0.5, 0.5])   # Green

        bbox_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]])

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
        ]

        colors = [[0.0, 0.0, 1.0] for _ in lines]  # Blue color for box lines

        bbox = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(lines))
        bbox.colors = o3d.utility.Vector3dVector(colors)

        env.vis.add_geometry(bbox)
        env.vis.add_geometry(env.pc1)
        env.vis.add_geometry(env.pc2)
        if corresp_transformed is not None and corresp_target is not None:
            env.vis.add_geometry(env.line_set)
        env.added = True
        env.vis.poll_events()
        env.vis.update_renderer()

    def update(env, pc_current, pc_target,corresp_transformed=None, corresp_target=None,action=None,mask_start=None,mask_target=None,mask= None):
        pc_current = pc_current.detach().cpu().numpy().squeeze(0)
        pc_target = pc_target.detach().cpu().numpy().squeeze(0)

        if not env.added:
            env.initialize(pc_current, pc_target)
            env.pivot_point = o3d.geometry.PointCloud()
            env.pivot_point.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
            env.pivot_point.paint_uniform_color([0, 0, 0])
            env.vis.add_geometry(env.pivot_point)
            env.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            env.sphere.paint_uniform_color([0, 0, 0])  # Black
            env.sphere.translate(np.array([0, 0, 0]))
            env.arrow = env.create_axis_arrow(np.asarray(env.pivot_point.points).reshape(3,1), np.array([0,0,0]), length=0.5)
            env.vis.add_geometry(env.sphere)
            env.vis.add_geometry(env.arrow)
        else:
            # Update geometry points in-place           
            env.pc1.points = o3d.utility.Vector3dVector(np.asarray(pc_current))
            env.pc2.points = o3d.utility.Vector3dVector(np.asarray(pc_target))
            env.pc1.paint_uniform_color([0.5, 0.5, 0.5])
            env.pc2.paint_uniform_color([0.5, 0.5, 0.5]) 
            if mask_start is not None and mask_target is not None:
                for i in range(len(mask_start)):                    
                    color1 = np.asarray(env.pc1.colors)
                    color1[mask_start[i][0,:]] = np.array([1,0,1])
                    env.pc1.colors = o3d.utility.Vector3dVector(color1)
                    color2 = np.asarray(env.pc2.colors)
                    color2[mask_target[i][0,:]] = np.array([0,1,0])
                    env.pc2.colors = o3d.utility.Vector3dVector(color2)
            if action is not None:
                for i in range(len(mask_start)):
                    base = i*7
                    # pivotx = 0 + action[base+1] * 0.01
                    # pivoty = 0 + action[base + 2] * 0.01
                    # pivotz = 0 + action[base + 3] * 0.01
                    pivotx = action[base+1]
                    pivoty = action[base + 2]
                    pivotz = action[base + 3]
                    pivot_point = np.array([[pivotx, pivoty, pivotz]])
                    env.pivot_point.points = o3d.utility.Vector3dVector(pivot_point)
                    # axis = [
                    # np.linspace(-1, 1, env.axis_steps)[action[base + 4]],
                    # np.linspace(-1, 1, env.axis_steps)[action[base + 5]],
                    # np.linspace(-1, 1, env.axis_steps)[action[base + 6]]]
                    axis = [action[base + 4], action[base + 5], action[base + 6]]
                    axis /= np.linalg.norm(axis) + 1e-8  # Normalize
                    env.sphere.translate(np.asarray(env.pivot_point.points).reshape(3,1),relative=False)
                    env.vis.remove_geometry(env.arrow, reset_bounding_box=False)
                    env.arrow = env.create_axis_arrow(np.asarray(env.pivot_point.points).reshape(3,1), axis, length=0.5)
                    env.vis.add_geometry(env.arrow, reset_bounding_box=False)
            else:
                env.pivot_point.points = o3d.utility.Vector3dVector()
            if corresp_transformed is not None and corresp_target is not None:
                all_points = np.vstack((corresp_transformed, corresp_target))
                lines = [[i, i + len(corresp_transformed)] for i in range(len(corresp_transformed))]
                np.random.seed(0)
                line_colors = [np.random.rand(3) for _ in range(len(lines))]
                env.line_set.points = o3d.utility.Vector3dVector(all_points)
                env.line_set.lines = o3d.utility.Vector2iVector(lines)
                env.line_set.colors = o3d.utility.Vector3dVector(line_colors)

            env.vis.update_geometry(env.pc1)
            env.vis.update_geometry(env.pc2)
            env.vis.update_geometry(env.pivot_point)
            if corresp_transformed is not None and corresp_target is not None:
                env.vis.update_geometry(env.line_set)
            env.vis.update_geometry(env.sphere)
            env.vis.update_geometry(env.arrow)
            env.vis.poll_events()
            env.vis.update_renderer()

    def close(env):
        env.vis.destroy_window()

    def create_axis_arrow(env, origin, direction, length=0.02, radius=0.0005, color=[0, 0, 1], arrow=None):
        """
        Create an Open3D arrow pointing in `direction` starting at `origin`.
        """
    # Translate to origin
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        end = origin + direction * length
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius*10,
            cone_radius=radius*10,
            cylinder_height=length,
            cone_height=length/10
        )
        arrow.paint_uniform_color(color)

        # Rotate arrow from +Z (default) to desired direction
        default_dir = np.array([0, 0, 1])
        rot_axis = np.cross(default_dir, direction)
        if np.linalg.norm(rot_axis) < 1e-6:
            rot_matrix = np.eye(3) if np.dot(default_dir, direction) > 0 else -np.eye(3)
        else:
            rot_angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
            rot_matrix = R.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)).as_matrix()
        arrow.rotate(rot_matrix, center=(0,0,0))

        # Translate to origin
        arrow.translate(origin, relative=False)
        return arrow



import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class decode_and_visualize_action():
    def __init__(self, env):
        self.env = env
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="RL Alignment Viewer", width=800, height=600)
        self.pc_vis = o3d.geometry.PointCloud()
        self.pc_vis = o3d.utility.Vector3dVector(np.asarray(self.env.pc_current).copy())
        self.pc_vis.paint_uniform_color([0.5, 0.5, 0.5])
        colors = np.asarray(self.pc_vis.colors)
        colors[env.mobile_masks[0]] = np.array([1,0,1])
        self.pc_vis.colors = o3d.utility.Vector3dVector(colors)
        self.env.pc_target.paint_uniform_color([0.5, 0.5, 0.5])
        colors_target = np.asarray(self.env.pc_target.colors)
        colors_target[env.target_masks[1]] = np.array([0,1,0])
        self.env.pc_target.colors = o3d.utility.Vector3dVector(colors_target)


    # self.pc_vis.colors = o3d.utility.Vector3dVector(np.asarray(env.pc_start.colors).copy())

        
    
    def update(self,action):

        print("Decoded Action Info:")
        self.action = action
        points = np.asarray(self.pc_vis)
        for i in range(self.env.num_joints):
            geometries = [self.pc_vis]
            base = i * 7
            pivotx = -0.5 + action[base] * 0.01
            pivoty = -0.5 + action[base + 1] * 0.01
            pivotz = -0.5 + action[base + 2] * 0.01
            pivot_point = np.array([pivotx, pivoty, pivotz])

            # Decode axis components
            axis = [
                    np.linspace(-1, 1, self.env.axis_steps)[self.action[base + 3]],
                    np.linspace(-1, 1, self.env.axis_steps)[self.action[base + 4]],
                    np.linspace(-1, 1, self.env.axis_steps)[self.action[base + 5]],
                ]
            axis /= np.linalg.norm(axis) + 1e-8  # Normalize
            # axis = np.array([0,0,1])


            # Decode angle
            angle = np.linspace(-np.pi, np.pi, self.env.angle_steps)[self.action[base+6]]
            # angle_increments = np.arange(0,angle,step = 0.01*np.abs(angle)/angle)
            mask = self.env.mobile_masks[i]
            joint_points = points[mask].copy()

            print(f"\nJoint {i}:")
            print(f"  Pivot Point: {pivot_point}")
            print(f"  Rotation Axis: {axis}")
            print(f"  Rotation Angle: {angle:.3f} rad")

            if i < self.env.num_joints - 1:
                # Create a combined mask of all downstream joints
                downstream_mask = np.zeros_like(self.env.mobile_masks[0], dtype=bool)
                for j in range(i + 1, self.env.num_joints):
                    downstream_mask |= self.env.mobile_masks[j]

                # Apply rotation to all downstream points
                downstream_points = points[downstream_mask]
            


            # Apply rotation
            rot_mat = R.from_rotvec(angle * axis).as_matrix()
            rotated = ((joint_points - pivot_point) @ rot_mat.T) + pivot_point
            points[mask] = rotated

            if i < self.env.num_joints - 1:
                transformed_points = ((downstream_points - pivot_point) @ rot_mat.T) + pivot_point

                # Update only the downstream points in the current point cloud
                points[downstream_mask] = transformed_points

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.paint_uniform_color([0, 0, 0])  # Black
            sphere.translate(pivot_point)
            geometries.append(sphere)
            arrow = create_axis_arrow(pivot_point, axis, length=0.5)
            geometries.append(arrow)
            geometries.append(self.env.pc_target)
            self.pc_vis = o3d.utility.Vector3dVector(points)
            geometries[0] = self.pc_vis

            for u in range(0,4):
                self.vis.add_geometry(geometries[u])

            self.vis.poll_events()           
            self.vis.update_renderer()

        for i in range(10000):
            self.vis.poll_events()
            self.vis.update_renderer()
        # self.vis.destroy_window()



def create_axis_arrow(origin, direction, length=0.02, radius=0.0005, color=[0, 0, 1]):
    """
    Create an Open3D arrow pointing in `direction` starting at `origin`.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    end = origin + direction * length

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius*10,
        cone_radius=radius*10,
        cylinder_height=length,
        cone_height=length/10
    )
    arrow.paint_uniform_color(color)

    # Rotate arrow from +Z (default) to desired direction
    default_dir = np.array([0, 0, 1])
    rot_axis = np.cross(default_dir, direction)
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_matrix = np.eye(3) if np.dot(default_dir, direction) > 0 else -np.eye(3)
    else:
        rot_angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
        rot_matrix = R.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)).as_matrix()
    arrow.rotate(rot_matrix, center=(0, 0, 0))

    # Translate to origin
    arrow.translate(origin)
    return arrow