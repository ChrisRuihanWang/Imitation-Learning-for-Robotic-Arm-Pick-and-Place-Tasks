from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.managers import SceneEntityCfg
from pick_and_place_project.tasks.mdp.actions import FrankaGripperActionCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg

# ✅ 你之前加的 width 观测函数（别删）
from pick_and_place_project.tasks.mdp import observation as my_obs


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ActionsCfg:
    arm_action: DifferentialInverseKinematicsActionCfg | None = None
    gripper_action: FrankaGripperActionCfg | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # robot arm joints
        joint_pos_rel: ObsTerm = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel: ObsTerm = ObsTerm(func=mdp.joint_vel_rel)

        # ✅ gripper joints (2 dims)
        gripper_joint_pos_rel: ObsTerm = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                )
            },
        )
        gripper_joint_vel_rel: ObsTerm = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                )
            },
        )

        # ✅ width (1 dim) 让 policy “看得见抓没抓住”
        gripper_width: ObsTerm = ObsTerm(
            func=my_obs.franka_gripper_width,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImagesCfg(ObsGroup):
        # 顶视相机
        rgb: ObsTerm = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
            },
        )
        # 手腕相机
        wrist_rgb: ObsTerm = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    images: ImagesCfg = ImagesCfg()


@configclass
class SceneCfg(InteractiveSceneCfg):
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    decimation: int = 2
    scene: SceneCfg = SceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = None
        self.rewards = None

        # ground
        self.scene.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        # ✅ 你之前加过的 dome_light（wrist 很需要）
        self.scene.dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=1000.0,      # 你之前 wrist 暗，这里直接用更亮
                color=(1.0, 1.0, 1.0),
            ),
        )

        # robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # cube
        cube_size = (0.04, 0.04, 0.04)
        cube_z = cube_size[2] / 2.0 + 0.005
        self.scene.cube_0 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_0",
            spawn=sim_utils.CuboidCfg(
                size=cube_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, cube_z),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # basket
        basket_size = (0.25, 0.25, 0.10)
        basket_z = basket_size[2] / 2.0
        self.scene.basket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            spawn=sim_utils.CuboidCfg(
                size=basket_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.2, 1.0),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.75, 0.0, basket_z),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # actions: IK + gripper
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107],
            ),
        )

        self.actions.gripper_action = FrankaGripperActionCfg(
            asset_name="robot",
            joint_names=("panda_finger_joint1", "panda_finger_joint2"),
            open_pos=0.04,
            close_pos=0.0,
        )

        # ---------- 顶视相机 ----------
        sim_utils.create_prim("/World/OverheadCameraBase", "Xform")
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/OverheadCamera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.6, 0.0, 1.2),
                rot=(0.0, 1.0, 0.0, 0.0),   # 对应 W=1,X=0,Y=0,Z=0
                convention="ros",
            ),
        )

        # ---------- 手腕相机（挂在 panda_hand 上） ----------
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.00, 0.08),           # 你后面想调的就调这里
                rot=(0.0, 0.0, 0.0, 1.0),         # 先默认
                convention="ros",
            ),
        )

        print("==== DEBUG ACTION TERMS ====")
        for name, term_cfg in self.actions.__dict__.items():
            if term_cfg is None:
                print(name, "=> None")
                continue
            ct = getattr(term_cfg, "class_type", "<no class_type>")
            print(name, "=>", type(term_cfg), "class_type:", ct)
        print("==== END DEBUG ACTION TERMS ====")


@configclass
class PickPlaceEnvCfg_PLAY(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5