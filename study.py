




# Core imports for setup, simulation, training, evaluation, and packaging.
import os, sys, subprocess, time, math, random, json, shutil, warnings, textwrap, heapq
from pathlib import Path




# 0) Install only missing dependencies and verify PyBullet before starting the expensive experiment.
print("\n" + "="*100)
print("[00/18] Fast dependency setup — binary PyBullet only, no source compilation")
print("="*100, flush=True)

import importlib.util
import platform


def _present(import_name):
    try:
        return importlib.util.find_spec(import_name) is not None
    except Exception:
        return False


def _run_pip(args, label):
    """Run pip visibly. No quiet mode so Colab never appears frozen."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-input",
        *args,
    ]
    print(f"\n📦 {label}", flush=True)
    print("   $ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print("   pip | " + line, flush=True)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"pip failed for {label!r} (exit code {code})")
    print(f"   ✅ finished in {time.time()-t0:.1f}s", flush=True)


print("Python        :", sys.version.split()[0], flush=True)
print("Platform      :", platform.platform(), flush=True)
print("Machine       :", platform.machine(), flush=True)
print("PyBullet found:", _present("pybullet"), flush=True)








if not _present("pybullet"):
    pyver = sys.version_info[:2]
    machine = platform.machine().lower()
    system = platform.system().lower()

    if pyver <= (3, 11):
        _run_pip(
            ["--only-binary=:all:", "pybullet==3.2.7"],
            "Installing official pybullet 3.2.7 binary wheel",
        )
    elif pyver in ((3, 12), (3, 13)) and system == "linux" and machine in ("x86_64", "amd64"):
        _run_pip(
            ["--only-binary=:all:", "pybullet-arm64==3.2.8"],
            "Installing pybullet-arm64 3.2.8 drop-in binary wheel for modern Colab",
        )
    else:
        raise RuntimeError(
            "No supported prebuilt PyBullet wheel path was selected for this runtime.\n"
            f"Detected Python {pyver[0]}.{pyver[1]}, OS={system}, machine={machine}.\n"
            "Use a Colab Python 3.11, 3.12, or 3.13 Linux x86_64 runtime.\n"
            "The experiment intentionally refuses to compile PyBullet from source."
        )


REQUIREMENTS = [
    ("gymnasium",          "gymnasium"),
    ("imageio",            "imageio"),
    ("imageio-ffmpeg",     "imageio_ffmpeg"),
    ("pandas",             "pandas"),
    ("scipy",              "scipy"),
    ("tqdm",               "tqdm"),
    ("tabulate",           "tabulate"),
    ("matplotlib",         "matplotlib"),
    ("Pillow",             "PIL"),
    
    
    ("timm",               "timm"),
    ("opencv-python-headless", "cv2"),
    ("torchvision",        "torchvision"),
]

missing = [req for req, imp in REQUIREMENTS if not _present(imp)]
if missing:
    _run_pip(missing, "Installing only missing lightweight dependencies")
else:
    print("\n✅ All lightweight dependencies are already present.", flush=True)


print("\n🔬 Verifying PyBullet DIRECT server + bundled data...", flush=True)
try:
    import pybullet as _pb_verify
    import pybullet_data as _pb_data_verify
    _cid = _pb_verify.connect(_pb_verify.DIRECT)
    if _cid < 0:
        raise RuntimeError("pb.connect(pb.DIRECT) returned a negative client id")
    _pb_verify.setAdditionalSearchPath(_pb_data_verify.getDataPath())
    _plane = _pb_verify.loadURDF("plane.urdf")
    if _plane < 0:
        raise RuntimeError("Failed to load bundled plane.urdf")
    _pb_verify.disconnect(_cid)
    print("✅ PyBullet import, DIRECT connection, and URDF loading all work.", flush=True)
except Exception as exc:
    raise RuntimeError(f"PyBullet verification failed: {exc!r}") from exc

print("\n[00/18] ✅ Dependency stage complete.\n", flush=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym
from gymnasium import spaces

import pybullet as pb
import pybullet_data
from pybullet_utils import bullet_client

import imageio.v2 as imageio
from PIL import Image
from IPython.display import display, Video

warnings.filterwarnings("ignore")






# 1) One configuration block keeps robot limits, safety thresholds, training budgets, and evaluation settings consistent across every method.
class CFG:
    SEED = 42
    FAST_SINGLE_ROOM = True

    
    MAX_TRAIN_STEPS = 10_000
    MIN_TRAIN_STEPS = 7_000
    EARLY_STOP_WINDOW = 16
    EARLY_STOP_SUCCESS = 0.70
    EARLY_STOP_HUMAN_COLLISION = 0.10
    EVAL_EPISODES_PER_CASE = 6
    TARGET_WALLCLOCK_MIN = 88  

    
    
    TRAIN_CNN = True
    TRAIN_VIT = True
    TRAIN_MDTA = True
    TRAIN_LP_MDTA_PPO = True
    TRAIN_PROPOSED = True

    ROOM_X = 7.5
    ROOM_Y = 6.5
    PHYSICS_HZ = 120
    CONTROL_SUBSTEPS = 6
    MAX_EPISODE_STEPS = 240
    MAX_HUMANS = 5

    CAM_W = 96
    CAM_H = 72
    CAM_FOV = 82.0
    CAMERA_HEIGHT_M = 0.55
    HISTORY = 3
    MIDAS_STRIDE = 8

    
    
    
    
    
    
    
    
    
    
    GOAL_ASSIST = True
    GOAL_ASSIST_END_FRAC = 0.55
    GOAL_ASSIST_ALPHA_START = 0.72
    GOAL_ASSIST_ALPHA_END = 0.08
    GOAL_ASSIST_FORWARD_MIN = 0.22
    GOAL_ASSIST_FORWARD_MAX = 0.68
    GOAL_ASSIST_TURN_GAIN = 1.35
    GOAL_ASSIST_HAZARD_LIMIT = 0.68
    GOAL_ASSIST_MIN_CHANGE = 0.025
    ASSIST_BC_COEF = 0.08
    FREEZE_SPEED_MPS = 0.035
    FREEZE_PATIENCE_STEPS = 10
    FREEZE_REWARD_PENALTY = 0.10

    ROBOT_HEIGHT_M = 0.60
    ROBOT_RADIUS = 0.25
    MAX_LINEAR = 0.70
    MAX_ANGULAR = 1.9
    WHEEL_RADIUS = 0.10
    TRACK_WIDTH = 0.38

    
    DRIVE_SIGN = 1.0
    TURN_SIGN = 1.0

    HUMAN_RADIUS = 0.23
    PERSONAL_CLEARANCE = 0.60
    HARD_CLEARANCE = 0.28

    ROLLOUT = 128
    EPOCHS = 3
    MINIBATCH = 32
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP = 0.20
    LR = 3e-4
    VALUE_COEF = 0.50
    COST_VALUE_COEF = 0.25
    ENTROPY_COEF = 0.010
    AUX_COEF = 0.030
    MAX_GRAD_NORM = 0.7

    COST_LIMIT = 0.045
    DUAL_LR = 0.05
    MAX_LAGRANGE = 15.0

    USE_DEPTH_SHIELD = True
    SHIELD_THRESHOLD = 0.82

    SAVE_DEMO_VIDEO = False  
    VIDEO_FPS = 20
    RESULTS_ROOT = "/content/LP_MDTA_SINGLE_ROOM_GOAL_ASSIST_FINAL_V9"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(CFG.RESULTS_ROOT)
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = torch.device(CFG.DEVICE)
T0 = time.time()

def stamp(): return time.strftime("%H:%M:%S")
def stage(i, message):
    elapsed=time.time()-T0
    print("\n"+"="*100)
    print(f"[{i:02d}/18 | {stamp()} | elapsed {elapsed/60:.1f} min] {message}")
    print("="*100, flush=True)

print(f"Device: {DEVICE}")
print("Experiment: FINAL V9 — AUTO-CALIBRATED GOAL-DIRECTED SOCIAL NAVIGATION")
print(f"Maximum training budget per learned method: {CFG.MAX_TRAIN_STEPS:,} steps")
print(f"Early stop after >= {CFG.MIN_TRAIN_STEPS:,} steps if recent success >= {CFG.EARLY_STOP_SUCCESS:.0%}")
print(f"Evaluation episodes per named test case: {CFG.EVAL_EPISODES_PER_CASE}")
print(f"Wall-clock target: < {CFG.TARGET_WALLCLOCK_MIN} min on Tesla T4-class runtime (target, not guarantee)")
print("V9 final: AUTO actuator calibration + goal assist + actor masking + vectorized DWA + env reuse + fast video")
if DEVICE.type != "cuda":
    print("WARNING: GPU is strongly recommended; CPU will be much slower for MiDaS + LP-MDTA.")
else:
    try: print("GPU:", torch.cuda.get_device_name(0))
    except Exception: pass




stage(1, "Writing mathematical specification and literature-context metadata")

# 2) Save the mathematical method specification beside the results so the reported research can be traced back to the implementation.
THEORY = r"""
# LP-MDTA-RCPPO
## Log-Polar Multi-Dconv Transposed Attention + Risk-Constrained PPO

Observation at time t:

    o_t = { [I_(t-k), D_(t-k)] }_(k=0)^(K-1), g_t, a_(t-1)

where I is RGB, D = MiDaS(I) is relative monocular depth, K is a short temporal history, and g is an
external navigation-goal command. Simulator human coordinates, velocities,
segmentation, collision geometry and Bullet depth are NOT policy inputs.

### Log-polar retinal sampling

    r(u) = (exp(beta u)-1)/(exp(beta)-1),  u in [0,1]
    x = r(u) cos(theta)
    y = r(u) sin(theta)

The proposed encoder fuses a Cartesian branch and a log-polar/foveated branch.

### MDTA
For feature map X, layer normalization gives Y. Point-wise then depth-wise
convolutions create Q,K,V. For each head they are reshaped to C_h x HW and
normalized. Attention is channel covariance rather than spatial HW x HW:

    A = softmax( tau * Q K^T )
    Z = A V

This is inspired by Restormer's Multi-Dconv Head Transposed Attention (MDTA).

### Temporal predictive auxiliary objective

Per-frame visual embeddings e_1..e_K are passed through a temporal Transformer.
The penultimate temporal state predicts the newest visual embedding:

    L_pred = 1 - cos(P(z_(K-1)), stopgrad(e_K))

so recent motion structure is learned without pedestrian labels.

### Asymmetric social field
For human i with forward direction f_i and lateral direction s_i:

    d_f = (p_r-p_i)·f_i
    d_s = (p_r-p_i)·s_i

    phi_i = exp(-0.5[(d_f/sigma_f)^2 + (d_s/sigma_s)^2])

with sigma_front > sigma_back, penalizing cutting immediately in front of a
walking person more strongly.

### TTC risk
For relative position r and velocity v:

    TTC = -(r·v)/(||v||^2 + eps)

If TTC>0, predicted closest approach is r* = r + TTC v. Short TTC with small
closest approach produces additional safety cost.

### Task reward

    R_t = w_p Delta d_goal
          + w_g I(goal reached)
          - w_s C_social
          - w_t C_TTC
          - w_u ||a_t-a_(t-1)||^2
          - w_c I(collision)

### Safety cost

    C_t = 1.00 I(robot-human collision)
          +0.30 I(robot-static collision)
          +0.15 I(personal-space violation)
          +0.10 C_TTC

### Risk-constrained PPO

    rho_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)

    L_R = E[min(rho A_R, clip(rho,1-eps,1+eps) A_R)]
    L_C = E[rho A_C]

    maximize: L_R - lambda L_C

    lambda <- clip(lambda + eta(E[C]-C_limit), 0, lambda_max)

### Evidence standard
A simulation run can demonstrate empirical performance under specified test
conditions. It cannot mathematically prove collision-free real-world behavior,
especially with monocular relative depth. The code therefore reports collision,
social-distance, TTC, SPL, physics-penetration and confidence-interval metrics.
"""
(ROOT / "MATHEMATICAL_SPECIFICATION.md").write_text(THEORY)


# Published results below are context only, not a direct leaderboard, because they use different simulators and protocols.
LITERATURE_CONTEXT = pd.DataFrame([
    {
        "method": "ORCA / SFM / DWA",
        "year": "classical",
        "type": "classical social/local planning",
        "reported_result": "Used as classical baselines in 2025 social-navigation benchmark",
        "benchmark": "multi-scenario benchmark",
        "input_note": "typically privileged/map + agent state",
        "source": "Alyassi et al., Frontiers in Robotics and AI, 2025"
    },
    {
        "method": "DS-RNN",
        "year": 2021,
        "type": "structural recurrent RL (PPO)",
        "reported_result": "reported to outperform prior crowd-navigation methods; sim-to-real TurtleBot demo",
        "benchmark": "CrowdNav",
        "input_note": "human position/state representations",
        "source": "Liu et al., ICRA 2021"
    },
    {
        "method": "SoNIC",
        "year": 2025,
        "type": "uncertainty-aware constrained RL",
        "reported_result": "96.93% success; 4.5x fewer collisions than previous RL method",
        "benchmark": "standard CrowdNav",
        "input_note": "uncertainty/nonconformity augmented RL state",
        "source": "SoNIC-Social-Nav project / paper"
    },
    {
        "method": "ARPL",
        "year": 2026,
        "type": "HSA-SRNN + adaptive stabilized PPO + attraction/repulsion reward",
        "reported_result": "98.40% success; ~2% above compared SOTA; collision count reduced ~3x",
        "benchmark": "CrowdNav++",
        "input_note": "structured human/robot interaction state",
        "source": "Journal of King Saud University C&IS, 2026"
    },
    {
        "method": "Industrial DRL vs ORCA",
        "year": 2026,
        "type": "deep RL social navigation",
        "reported_result": "open 0.97 vs ORCA 1.00; fixed 0.82 vs 0.83; intersection 0.26 vs 0.15 success",
        "benchmark": "custom industrial scenarios",
        "input_note": "different simulator/protocol from this experiment",
        "source": "Procedia Computer Science / ScienceDirect, 2026"
    },
])
LITERATURE_CONTEXT.to_csv(ROOT / "literature_context_NOT_DIRECTLY_COMPARABLE.csv", index=False)
print(LITERATURE_CONTEXT.to_string(index=False))
print("\nIMPORTANT: external literature numbers are contextual only, not direct head-to-head scores.")




stage(2, "Loading MiDaS_small; Torch Hub may display its own download progress")

print("[MiDaS] Loading model...", flush=True)
# 3) MiDaS_small supplies relative monocular depth from the same RGB camera view available to the learned policies.
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True)
midas = midas.to(DEVICE).eval()
print("[MiDaS] Loading transforms...", flush=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)
midas_transform = midas_transforms.small_transform
for p in midas.parameters():
    p.requires_grad_(False)
print("[MiDaS] Ready.", flush=True)

# Normalize MiDaS output into a stable relative-depth map and measure inference time for the compute report.
class MidasDepth:
    def __init__(self):
        self.calls = 0
        self.total_time = 0.0

    @torch.inference_mode()
    def predict(self, rgb_uint8):
        t = time.time()
        
        
        img = rgb_uint8.astype(np.float32)
        batch = midas_transform(img).to(DEVICE)
        pred = midas(batch)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(CFG.CAM_H, CFG.CAM_W), mode="bicubic", align_corners=False)
        d = pred[0, 0].float().cpu().numpy()
        lo, hi = np.percentile(d, [3, 97])
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)
        self.calls += 1
        self.total_time += time.time() - t
        return d

depth_model = MidasDepth()




stage(3, "Generating the 0.60 m differential-drive Wall-E-like robot URDF")

# 4) Define the differential-drive robot inside the experiment so the physics setup is self-contained and reproducible.
ROBOT_URDF = r'''<?xml version="1.0"?>
<robot name="walle_social_robot">
  <material name="yellow"><color rgba="0.82 0.64 0.12 1"/></material>
  <material name="dark"><color rgba="0.10 0.10 0.10 1"/></material>

  <link name="base">
    <inertial><mass value="8.0"/><inertia ixx="0.10" ixy="0" ixz="0" iyy="0.12" iyz="0" izz="0.14"/></inertial>
    <visual><geometry><box size="0.40 0.34 0.28"/></geometry><material name="yellow"/></visual>
    <collision><geometry><box size="0.40 0.34 0.28"/></geometry></collision>
  </link>

  <link name="head">
    <inertial><mass value="1.4"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <visual><geometry><box size="0.30 0.24 0.20"/></geometry><material name="dark"/></visual>
    <collision><geometry><box size="0.30 0.24 0.20"/></geometry></collision>
  </link>
  <joint name="head_fixed" type="fixed"><parent link="base"/><child link="head"/><origin xyz="0.02 0 0.20"/></joint>

  <link name="left_wheel">
    <inertial><mass value="0.40"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><origin rpy="1.570796 0 0"/><geometry><cylinder radius="0.10" length="0.06"/></geometry><material name="dark"/></visual>
    <collision><origin rpy="1.570796 0 0"/><geometry><cylinder radius="0.10" length="0.06"/></geometry></collision>
  </link>
  <joint name="left_wheel_joint" type="continuous"><parent link="base"/><child link="left_wheel"/><origin xyz="0 0.19 -0.18"/><axis xyz="0 1 0"/></joint>

  <link name="right_wheel">
    <inertial><mass value="0.40"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><origin rpy="1.570796 0 0"/><geometry><cylinder radius="0.10" length="0.06"/></geometry><material name="dark"/></visual>
    <collision><origin rpy="1.570796 0 0"/><geometry><cylinder radius="0.10" length="0.06"/></geometry></collision>
  </link>
  <joint name="right_wheel_joint" type="continuous"><parent link="base"/><child link="right_wheel"/><origin xyz="0 -0.19 -0.18"/><axis xyz="0 1 0"/></joint>

  <link name="front_caster">
    <inertial><mass value="0.08"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><sphere radius="0.045"/></geometry><material name="dark"/></visual>
    <collision><geometry><sphere radius="0.045"/></geometry></collision>
  </link>
  <joint name="front_caster_joint" type="fixed"><parent link="base"/><child link="front_caster"/><origin xyz="0.17 0 -0.235"/></joint>

  <link name="rear_caster">
    <inertial><mass value="0.08"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><sphere radius="0.045"/></geometry><material name="dark"/></visual>
    <collision><geometry><sphere radius="0.045"/></geometry></collision>
  </link>
  <joint name="rear_caster_joint" type="fixed"><parent link="base"/><child link="rear_caster"/><origin xyz="-0.17 0 -0.235"/></joint>
</robot>'''

URDF_PATH = ROOT / "walle_social_robot.urdf"
URDF_PATH.write_text(ROBOT_URDF)
print(f"Robot URDF saved: {URDF_PATH}")






stage(4, "Defining one furnished room + named social-navigation test cases")

TEST_CASES = ["empty", "crossing_1", "head_on_1", "crossing_3", "dense_5", "sensor_noise"]
TRAIN_CASES_EASY = ["empty", "crossing_1", "head_on_1"]
TRAIN_CASES_MED = ["crossing_1", "head_on_1", "crossing_3"]
TRAIN_CASES_HARD = ["head_on_1", "crossing_3", "dense_5"]

# 5) Main physics environment: furnished room, moving humans, observations, social-risk metrics, reward, collisions, and episode termination.
class SingleRoomSocialEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed=42, scenario="train", sensor_mode="full", enable_depth_shield=True):
        super().__init__()
        self.sensor_mode = sensor_mode
        self.enable_depth_shield = bool(enable_depth_shield)
        self._zero_vision = np.zeros((CFG.HISTORY*4,CFG.CAM_H,CFG.CAM_W),np.float32)
        self.rng = np.random.default_rng(seed)
        self.requested_scenario = scenario
        self.scenario = scenario
        self.reset_count = 0
        self.p = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.dt = 1.0 / CFG.PHYSICS_HZ
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=100,
            deterministicOverlappingPairs=1, contactERP=0.25,
            frictionERP=0.20, numSubSteps=0)
        self.p.setGravity(0,0,-9.81)
        self.static_ids=set(); self.human_ids=[]; self.active_humans=[]
        self.wall_rects=[]; self.furniture_rects=[]
        self._build_world()

        self.observation_space=spaces.Dict({
            "vision":spaces.Box(0.0,1.0,shape=(CFG.HISTORY*4,CFG.CAM_H,CFG.CAM_W),dtype=np.float32),
            "cmd":spaces.Box(-1.0,1.0,shape=(5,),dtype=np.float32)})
        self.action_space=spaces.Box(-1.0,1.0,shape=(2,),dtype=np.float32)
        self.reset(seed=seed)

    def close(self):
        try: self.p.disconnect()
        except Exception: pass

    def _add_box(self,pos,half,rgba=(0.6,0.6,0.6,1),category=None):
        col=self.p.createCollisionShape(pb.GEOM_BOX,halfExtents=half)
        vis=self.p.createVisualShape(pb.GEOM_BOX,halfExtents=half,rgbaColor=rgba)
        bid=self.p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,
                                   baseVisualShapeIndex=vis,basePosition=pos)
        self.static_ids.add(bid)
        rect=(pos[0]-half[0],pos[1]-half[1],2*half[0],2*half[1])
        if category=="wall": self.wall_rects.append(rect)
        if category=="furniture": self.furniture_rects.append(rect)
        return bid

    def _build_world(self):
        self.p.resetSimulation(); self.p.setGravity(0,0,-9.81)
        self.plane=self.p.loadURDF("plane.urdf")
        self.p.changeDynamics(self.plane,-1,lateralFriction=.9,restitution=0)
        self.p.changeVisualShape(self.plane,-1,rgbaColor=[.79,.77,.72,1])

        hx,hy=CFG.ROOM_X/2,CFG.ROOM_Y/2; h=1.2; t=.065
        for pos,half in [([-hx,0,h],[t,hy,h]),([hx,0,h],[t,hy,h]),
                         ([0,-hy,h],[hx,t,h]),([0,hy,h],[hx,t,h])]:
            self._add_box(pos,half,(.88,.88,.90,1),"wall")

        furniture=[
            ([-2.20, 1.72,.34],[.48,.32,.34],(.35,.48,.65,1)),
            ([ 2.15, 1.72,.32],[.55,.30,.32],(.36,.52,.37,1)),
            ([-2.15,-1.72,.45],[.40,.32,.45],(.48,.34,.22,1)),
            ([ 2.20,-1.70,.50],[.42,.30,.50],(.62,.62,.66,1)),
            ([ 0.00, 1.90,.32],[.55,.22,.32],(.47,.30,.17,1)),
            ([ 0.45,-1.45,.32],[.30,.28,.32],(.42,.28,.17,1)),
        ]
        for pos,half,color in furniture:
            self._add_box(pos,half,color,"furniture")

        self.nodes=np.array([
            [-2.45,-1.4],[-2.45,0],[-2.45,1.4],
            [0,-1.7],[0,0],[0,1.7],
            [2.45,-1.4],[2.45,0],[2.45,1.4]],dtype=np.float32)

        self.robot=self.p.loadURDF(str(URDF_PATH),[0,0,.28],useFixedBase=False,
                                   flags=pb.URDF_USE_INERTIA_FROM_FILE)
        self.left_joint=self.right_joint=None
        for j in range(self.p.getNumJoints(self.robot)):
            nm=self.p.getJointInfo(self.robot,j)[1].decode()
            if nm=="left_wheel_joint": self.left_joint=j
            if nm=="right_wheel_joint": self.right_joint=j
            if "caster" in nm:
                self.p.changeDynamics(self.robot,j,lateralFriction=.03,spinningFriction=.001,rollingFriction=.001)
        self.p.changeDynamics(self.robot,self.left_joint,lateralFriction=1.35,spinningFriction=.003)
        self.p.changeDynamics(self.robot,self.right_joint,lateralFriction=1.35,spinningFriction=.003)

        hcol=self.p.createCollisionShape(pb.GEOM_CAPSULE,radius=CFG.HUMAN_RADIUS,height=1.22)
        colors=[(.85,.25,.20,1),(.20,.45,.85,1),(.30,.70,.35,1),(.75,.35,.75,1),(.85,.65,.15,1)]
        for i in range(CFG.MAX_HUMANS):
            vis=self.p.createVisualShape(pb.GEOM_CAPSULE,radius=CFG.HUMAN_RADIUS,length=1.22,rgbaColor=colors[i])
            hid=self.p.createMultiBody(baseMass=65,baseCollisionShapeIndex=hcol,
                                       baseVisualShapeIndex=vis,basePosition=[20+i,20,.84])
            self.p.changeDynamics(hid,-1,lateralFriction=.18,restitution=0,
                                  linearDamping=.25,angularDamping=.995)
            self.human_ids.append(hid)

    # Curriculum begins with easier goal reaching, then increases human-interaction difficulty; every learned method gets the same schedule.
    def _select_training_case(self):
        
        
        if self.reset_count < 24: pool=TRAIN_CASES_EASY
        elif self.reset_count < 52: pool=TRAIN_CASES_MED
        else: pool=TRAIN_CASES_HARD
        return str(self.rng.choice(pool))

    def _case_layout(self,case):
        jitter=lambda s: float(s*self.rng.uniform(.90,1.10))
        if case=="empty":
            return (-2.35,-.65),(2.35,.65),[]
        if case=="crossing_1":
            return (-2.35,-.75),(2.35,-.75),[
                ((0,-2.0),(0,2.0),jitter(.85))]
        if case=="head_on_1":
            return (-2.35,0),(2.35,0),[
                ((2.0,0),(-2.0,0),jitter(.78))]
        if case in ("crossing_3","sensor_noise"):
            return (-2.35,-.85),(2.35,.85),[
                ((-.85,-2.0),(-.85,2.0),jitter(.78)),
                (( .10, 2.0),( .10,-2.0),jitter(.88)),
                (( .95,-2.0),( .95,2.0),jitter(.72))]
        if case=="dense_5":
            return (-2.35,0),(2.35,0),[
                ((-1.05,-2.0),(-1.05,2.0),jitter(.80)),
                (( 0.00, 2.0),( 0.00,-2.0),jitter(.90)),
                (( 1.00,-2.0),( 1.00,2.0),jitter(.75)),
                (( 2.00,.75),(-2.00,.75),jitter(.72)),
                ((-1.80,-.80),(1.80,-.80),jitter(.70))]
        raise ValueError(case)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng=np.random.default_rng(seed)
        self.reset_count += 1
        self.scenario=self._select_training_case() if self.requested_scenario=="train" else self.requested_scenario

        self.t=0; self.prev_action=np.zeros(2,np.float32); self.depth_cache=None; self.frames=[]
        self.stationary_steps=0
        self.stationary_total=0
        self.forced_exploration_total=0
        self.trajectory=[]; self.human_trajectories=[[] for _ in self.human_ids]
        self.min_clearance=999.; self.discomfort_steps=0; self.social_cost_sum=0.; self.ttc_cost_sum=0.
        self.shield_count=0; self.path_length=0.; self.smoothness_sum=0.
        self.max_human_penetration=0.; self.ttc_risk_steps=0
        self.robot_human_collision=0; self.robot_static_collision=0
        self.success=0; self.terminal_reason="running"

        start,goal,hspec=self._case_layout(self.scenario)
        p0=np.asarray(start,np.float32)+self.rng.normal(0,.045,2)
        self.goal_xy=np.asarray(goal,np.float32)+self.rng.normal(0,.035,2)
        yaw=math.atan2(self.goal_xy[1]-p0[1],self.goal_xy[0]-p0[0])+float(self.rng.normal(0,.10))
        self.p.resetBasePositionAndOrientation(self.robot,[p0[0],p0[1],.28],
                                               self.p.getQuaternionFromEuler([0,0,yaw]))
        self.p.resetBaseVelocity(self.robot,[0,0,0],[0,0,0])
        for j in (self.left_joint,self.right_joint):
            self.p.resetJointState(self.robot,j,0,0)
        self.last_goal_dist=float(np.linalg.norm(self.goal_xy-p0))
        self.shortest_path_m=self.last_goal_dist
        self.trajectory.append(p0.copy())

        self.active_humans=[]; self.human_target=[]; self.human_origin=[]; self.human_speed=[]
        for i,hid in enumerate(self.human_ids):
            if i < len(hspec):
                st,tg,sp=hspec[i]
                st=np.asarray(st,np.float32)+self.rng.normal(0,.035,2)
                tg=np.asarray(tg,np.float32)
                self.p.resetBasePositionAndOrientation(hid,[st[0],st[1],.84],
                                                       self.p.getQuaternionFromEuler([0,0,0]))
                self.p.resetBaseVelocity(hid,[0,0,0],[0,0,0])
                self.active_humans.append(hid)
                self.human_origin.append(st.copy())
                self.human_target.append(tg.copy())
                self.human_speed.append(float(sp))
                self.human_trajectories[i].append(st.copy())
            else:
                self.p.resetBasePositionAndOrientation(hid,[20+i,20,.84],
                                                       self.p.getQuaternionFromEuler([0,0,0]))
                self.p.resetBaseVelocity(hid,[0,0,0],[0,0,0])

        for _ in range(12): self.p.stepSimulation()
        return self._get_obs(initialize=True),{"case":self.scenario}

    def _human_controller(self):
        if not self.active_humans: return
        robot=np.asarray(self.p.getBasePositionAndOrientation(self.robot)[0][:2],np.float32)
        positions=[np.asarray(self.p.getBasePositionAndOrientation(h)[0][:2],np.float32)
                   for h in self.active_humans]
        for i,hid in enumerate(self.active_humans):
            pos=positions[i]; target=self.human_target[i]
            delta=target-pos; dist=float(np.linalg.norm(delta))
            if dist<.22:
                old=self.human_origin[i].copy()
                self.human_origin[i]=self.human_target[i].copy()
                self.human_target[i]=old
                target=self.human_target[i]; delta=target-pos; dist=float(np.linalg.norm(delta))
            desired=delta/(dist+1e-6)*self.human_speed[i]
            rep=np.zeros(2,np.float32)
            for j,pj in enumerate(positions):
                if i==j: continue
                dv=pos-pj; d=float(np.linalg.norm(dv))
                if d<.85: rep += dv/(d+1e-5)*(2.4*(.85-d))
            dv=pos-robot; d=float(np.linalg.norm(dv))
            if d<1.0: rep += dv/(d+1e-5)*(1.5*(1.0-d))
            desired += rep
            n=float(np.linalg.norm(desired))
            if n>1.35: desired*=1.35/n
            cur=np.asarray(self.p.getBaseVelocity(hid)[0][:2],np.float32)
            force=210*(desired-cur); fn=float(np.linalg.norm(force))
            if fn>250: force*=250/fn
            self.p.applyExternalForce(hid,-1,[float(force[0]),float(force[1]),0],
                                      [0,0,0],pb.WORLD_FRAME)

    def _stabilize_humans(self):
        for hid in self.active_humans:
            pos,quat=self.p.getBasePositionAndOrientation(hid)
            vel,_=self.p.getBaseVelocity(hid)
            yaw=math.atan2(vel[1],vel[0]) if math.hypot(vel[0],vel[1])>.08 else self.p.getEulerFromQuaternion(quat)[2]
            self.p.resetBasePositionAndOrientation(hid,[pos[0],pos[1],.84],
                                                   self.p.getQuaternionFromEuler([0,0,yaw]))
            self.p.resetBaseVelocity(hid,[vel[0],vel[1],0],[0,0,0])

    def _camera_rgb(self):
        pos,quat=self.p.getBasePositionAndOrientation(self.robot)
        yaw=self.p.getEulerFromQuaternion(quat)[2]
        eye=[pos[0]+.15*math.cos(yaw),pos[1]+.15*math.sin(yaw),CFG.CAMERA_HEIGHT_M]
        target=[eye[0]+math.cos(yaw),eye[1]+math.sin(yaw),eye[2]-.02]
        view=self.p.computeViewMatrix(eye,target,[0,0,1])
        proj=self.p.computeProjectionMatrixFOV(CFG.CAM_FOV,CFG.CAM_W/CFG.CAM_H,.05,7.0)
        im=self.p.getCameraImage(CFG.CAM_W,CFG.CAM_H,view,proj,renderer=pb.ER_TINY_RENDERER)
        rgb=np.asarray(im[2],np.uint8).reshape(CFG.CAM_H,CFG.CAM_W,4)[:,:,:3]
        if self.scenario=="sensor_noise":
            rgb=np.clip(rgb.astype(np.float32)+self.rng.normal(0,7.0,rgb.shape),0,255).astype(np.uint8)
        self.last_rgb=rgb
        return rgb

    def render_overhead(self,W=640,H=500):
        view=self.p.computeViewMatrix([0,0,9],[0,0,0],[0,1,0])
        proj=self.p.computeProjectionMatrixFOV(42,W/H,.1,15)
        im=self.p.getCameraImage(W,H,view,proj,renderer=pb.ER_TINY_RENDERER)
        return np.asarray(im[2],np.uint8).reshape(H,W,4)[:,:,:3]

    def _goal_command(self):
        pos,quat=self.p.getBasePositionAndOrientation(self.robot)
        xy=np.asarray(pos[:2],np.float32); yaw=self.p.getEulerFromQuaternion(quat)[2]
        dxy=self.goal_xy-xy; dist=float(np.linalg.norm(dxy))
        desired=math.atan2(dxy[1],dxy[0])
        err=(desired-yaw+math.pi)%(2*math.pi)-math.pi
        return np.array([np.clip(dist/5.5,0,1),math.sin(err),math.cos(err),
                         self.prev_action[0],self.prev_action[1]],np.float32)

    # Learned policies receive RGB + MiDaS history and the egocentric goal command. Non-visual baselines skip rendering only to reduce evaluation cost.
    def _get_obs(self,initialize=False):
        
        
        
        
        if self.sensor_mode in ("command_only","privileged"):
            return {"vision":self._zero_vision, "cmd":self._goal_command()}
        rgb=self._camera_rgb()
        if self.depth_cache is None or self.t%CFG.MIDAS_STRIDE==0:
            self.depth_cache=depth_model.predict(rgb)
        depth=self.depth_cache.copy()
        if self.scenario=="sensor_noise":
            depth=np.clip(depth+self.rng.normal(0,.03,depth.shape),0,1).astype(np.float32)
        frame=np.concatenate([(rgb.astype(np.float32)/255.).transpose(2,0,1),
                              depth[None]],0).astype(np.float32)
        if initialize:
            self.frames=[frame.copy() for _ in range(CFG.HISTORY)]
        else:
            self.frames.append(frame)
            if len(self.frames)>CFG.HISTORY: self.frames.pop(0)
        return {"vision":np.concatenate(self.frames,0).astype(np.float32),
                "cmd":self._goal_command()}

    # The depth shield can only reduce unsafe forward motion; it does not choose the route or provide privileged human state.
    def _depth_shield(self,action):
        action=np.clip(np.asarray(action,np.float32),-1,1)
        if (not self.enable_depth_shield) or (not CFG.USE_DEPTH_SHIELD) or self.depth_cache is None or action[0]<=0:
            return action,False,0.
        H,W=self.depth_cache.shape
        crop=self.depth_cache[int(.38*H):int(.92*H),int(.34*W):int(.66*W)]
        hazard=float(np.quantile(crop,.90))
        trig=hazard>CFG.SHIELD_THRESHOLD
        if trig:
            alpha=np.clip((1-hazard)/(1-CFG.SHIELD_THRESHOLD+1e-6),0,1)
            action[0]*=alpha**2
        return action,trig,hazard

    # Convert normalized forward/turn commands into wheel speeds using actuator signs measured automatically before training.
    def _drive(self,action):
        
        
        
        
        v=float(action[0]*CFG.MAX_LINEAR)
        w=float(action[1]*CFG.MAX_ANGULAR)
        wl=CFG.DRIVE_SIGN*(v-CFG.TURN_SIGN*w*CFG.TRACK_WIDTH*.5)/CFG.WHEEL_RADIUS
        wr=CFG.DRIVE_SIGN*(v+CFG.TURN_SIGN*w*CFG.TRACK_WIDTH*.5)/CFG.WHEEL_RADIUS
        self.p.setJointMotorControl2(
            self.robot,self.left_joint,pb.VELOCITY_CONTROL,
            targetVelocity=wl,force=5.5)
        self.p.setJointMotorControl2(
            self.robot,self.right_joint,pb.VELOCITY_CONTROL,
            targetVelocity=wr,force=5.5)

    # Compute personal-space and TTC risk for reward/evaluation. These simulator-only quantities are not included in learned-policy observations.
    def _social_metrics(self):
        rpos=np.asarray(self.p.getBasePositionAndOrientation(self.robot)[0][:2],np.float32)
        rvel=np.asarray(self.p.getBaseVelocity(self.robot)[0][:2],np.float32)
        social=0.; ttc_cost=0.; min_clear=999.
        for hid in self.active_humans:
            hp=np.asarray(self.p.getBasePositionAndOrientation(hid)[0][:2],np.float32)
            hv=np.asarray(self.p.getBaseVelocity(hid)[0][:2],np.float32)
            rel=rpos-hp; cd=float(np.linalg.norm(rel))
            clear=max(0.,cd-(CFG.ROBOT_RADIUS+CFG.HUMAN_RADIUS))
            min_clear=min(min_clear,clear)
            speed=float(np.linalg.norm(hv))
            forward=hv/(speed+1e-6) if speed>.12 else np.array([1.,0.])
            side=np.array([-forward[1],forward[0]])
            df,ds=float(rel@forward),float(rel@side)
            sigma_f=1.15 if df>=0 else .68
            social=max(social,math.exp(-.5*((df/sigma_f)**2+(ds/.70)**2)))
            rp,rv=hp-rpos,hv-rvel; vv=float(rv@rv)
            if vv>1e-4:
                ttc=-float(rp@rv)/(vv+1e-6)
                if 0<ttc<2.5:
                    closest=rp+ttc*rv; c=float(np.linalg.norm(closest))
                    if c<.95:
                        ttc_cost=max(ttc_cost,float(((2.5-ttc)/2.5)*((.95-c)/.95)))
        if not self.active_humans: min_clear=9.0
        return float(social),float(ttc_cost),float(min_clear)

    def _collision_metrics(self):
        rh=False; rs=False
        for cp in self.p.getContactPoints(bodyA=self.robot):
            other=cp[2]
            if other in self.active_humans: rh=True
            elif other in self.static_ids and other!=self.plane: rs=True
        max_pen=0.
        for i in range(len(self.active_humans)):
            for j in range(i+1,len(self.active_humans)):
                for cp in self.p.getClosestPoints(self.active_humans[i],self.active_humans[j],distance=0.0):
                    if cp[8]<0: max_pen=max(max_pen,-float(cp[8]))
        return rh,rs,max_pen

    def static_clearance_xy(self,xy):
        x,y=float(xy[0]),float(xy[1]); best=999.
        for rx,ry,rw,rh in self.wall_rects+self.furniture_rects:
            cx=np.clip(x,rx,rx+rw); cy=np.clip(y,ry,ry+rh)
            d=math.hypot(x-cx,y-cy)
            inside=rx<=x<=rx+rw and ry<=y<=ry+rh
            if inside: d=-min(x-rx,rx+rw-x,y-ry,ry+rh-y)
            best=min(best,float(d))
        return best

    def privileged_state(self):
        rpos,quat=self.p.getBasePositionAndOrientation(self.robot)
        yaw=self.p.getEulerFromQuaternion(quat)[2]
        humans=[]
        for hid in self.active_humans:
            hp=np.asarray(self.p.getBasePositionAndOrientation(hid)[0][:2],np.float32)
            hv=np.asarray(self.p.getBaseVelocity(hid)[0][:2],np.float32)
            humans.append((hp,hv))
        return np.asarray(rpos[:2],np.float32),float(yaw),self.goal_xy.copy(),humans

    # Run one control step: apply safety gating, advance robot/humans, calculate reward/costs, and check goal/collision/timeout termination.
    def step(self,action):
        action,shielded,hazard=self._depth_shield(action)
        if shielded: self.shield_count+=1
        prev=np.asarray(self.p.getBasePositionAndOrientation(self.robot)[0][:2],np.float32)
        self._drive(action)
        for _ in range(CFG.CONTROL_SUBSTEPS):
            self._human_controller(); self.p.stepSimulation(); self._stabilize_humans()

        self.t+=1
        pos=np.asarray(self.p.getBasePositionAndOrientation(self.robot)[0][:2],np.float32)
        displacement=float(np.linalg.norm(pos-prev))
        self.path_length+=displacement
        inst_speed=displacement*CFG.PHYSICS_HZ/max(CFG.CONTROL_SUBSTEPS,1)
        if inst_speed < CFG.FREEZE_SPEED_MPS:
            self.stationary_steps += 1
            self.stationary_total += 1
        else:
            self.stationary_steps = 0
        self.trajectory.append(pos.copy())
        for i,hid in enumerate(self.human_ids):
            if hid in self.active_humans:
                self.human_trajectories[i].append(
                    np.asarray(self.p.getBasePositionAndOrientation(hid)[0][:2],np.float32))

        dist=float(np.linalg.norm(self.goal_xy-pos))
        progress=self.last_goal_dist-dist
        social,ttc,minclear=self._social_metrics()
        rh,rs,hpen=self._collision_metrics()
        self.max_human_penetration=max(self.max_human_penetration,hpen)
        self.min_clearance=min(self.min_clearance,minclear)
        discomfort=minclear<CFG.PERSONAL_CLEARANCE
        if discomfort: self.discomfort_steps+=1
        if ttc>.20: self.ttc_risk_steps+=1
        self.social_cost_sum+=social; self.ttc_cost_sum+=ttc
        smooth=float(np.sum((action-self.prev_action)**2))
        self.smoothness_sum+=smooth

        cmd=self._goal_command()
        heading_alignment=max(0.,float(cmd[2]))
        forward=max(0.,float(action[0]))
        
        
        reward=(12.0*progress + .060*forward*heading_alignment
                - .36*social - .56*ttc - .010*smooth - .006)
        
        
        
        if self.stationary_steps > CFG.FREEZE_PATIENCE_STEPS:
            reward -= CFG.FREEZE_REWARD_PENALTY
        if minclear<CFG.HARD_CLEARANCE:
            reward-=1.1*(CFG.HARD_CLEARANCE-minclear)/CFG.HARD_CLEARANCE

        reached=dist<.38
        terminated=False
        
        
        if rh:
            reward-=20.
            self.robot_human_collision=1
            self.terminal_reason="human_collision"
            terminated=True
        elif rs:
            reward-=8.
            self.robot_static_collision=1
            self.terminal_reason="static_collision"
            terminated=True
        elif reached:
            reward+=22.
            self.success=1
            self.terminal_reason="success"
            terminated=True

        safety=float(rh)+.30*float(rs)+.15*float(discomfort)+.10*ttc
        truncated=self.t>=CFG.MAX_EPISODE_STEPS and not terminated
        if truncated:
            self.terminal_reason="timeout"
            reward -= 1.0

        self.last_goal_dist=dist
        self.prev_action=action.astype(np.float32)
        obs=self._get_obs(False)
        info={"case":self.scenario,"social_cost":social,"ttc_cost":ttc,
              "clearance":minclear,"hazard":hazard,"shield":float(shielded),
              "safety_cost":safety,"human_penetration":hpen}

        if terminated or truncated:
            spl=self.success*self.shortest_path_m/max(self.path_length,self.shortest_path_m,1e-6)
            duration=self.t*CFG.CONTROL_SUBSTEPS/CFG.PHYSICS_HZ
            info["episode"]={
                "success":int(self.success),
                "human_collision":int(self.robot_human_collision),
                "static_collision":int(self.robot_static_collision),
                "timeout":int(self.terminal_reason=="timeout"),
                "terminal_reason":self.terminal_reason,
                "spl":float(spl),
                "shortest_path_m":float(self.shortest_path_m),
                "path_length_m":float(self.path_length),
                "min_clearance_m":float(self.min_clearance),
                "discomfort_fraction":float(self.discomfort_steps/max(self.t,1)),
                "ttc_risk_fraction":float(self.ttc_risk_steps/max(self.t,1)),
                "mean_social_cost":float(self.social_cost_sum/max(self.t,1)),
                "mean_ttc_cost":float(self.ttc_cost_sum/max(self.t,1)),
                "smoothness":float(self.smoothness_sum/max(self.t,1)),
                "shield_fraction":float(self.shield_count/max(self.t,1)),
                "max_human_penetration_m":float(self.max_human_penetration),
                "steps":int(self.t),
                "time_s":float(duration),
                "case":self.scenario,
                "mean_speed_proxy":float(self.path_length/max(duration,1e-6)),
                "stationary_fraction":float(self.stationary_total/max(self.t,1))}
        return obs,float(reward),bool(terminated),bool(truncated),info





stage(5, "Running single-room physics + sensor + named-case self-tests")
# 6) Self-test the environment before training so broken observations or unstable human physics fail early.
for case in ["empty","crossing_1","dense_5"]:
    env=SingleRoomSocialEnv(seed=CFG.SEED,scenario=case)
    obs,info=env.reset(seed=CFG.SEED)
    assert obs["vision"].shape==(CFG.HISTORY*4,CFG.CAM_H,CFG.CAM_W)
    maxpen=0.
    for _ in tqdm(range(16),desc=f"self-test {case}",leave=False):
        obs,r,te,tr,inf=env.step(np.array([.15,0],np.float32))
        maxpen=max(maxpen,float(inf["human_penetration"]))
        if te or tr: break
    print(f"[self-test] {case:12s} obs={obs['vision'].shape} active_humans={len(env.active_humans)} max_human_pen={maxpen:.4f} m")
    assert np.isfinite(obs["vision"]).all()
    assert maxpen < .05
    env.close()
print("[self-test] PASS")





print("\n" + "="*100)
print("[05B/18] Automatic wheel-sign calibration + deterministic goal-reaching preflight")
print("="*100)

def _wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def _forward_projection_for_sign(sign):
    CFG.DRIVE_SIGN=float(sign)
    CFG.TURN_SIGN=1.0
    env=SingleRoomSocialEnv(
        seed=CFG.SEED+7101,
        scenario="empty",
        sensor_mode="command_only",
        enable_depth_shield=False)
    try:
        env.reset(seed=CFG.SEED+7101)
        p0,q0=env.p.getBasePositionAndOrientation(env.robot)
        yaw0=env.p.getEulerFromQuaternion(q0)[2]
        heading=np.array([math.cos(yaw0),math.sin(yaw0)],np.float32)

        for _ in range(18):
            _,_,te,tr,_=env.step(np.array([0.45,0.0],np.float32))
            if te or tr: break

        p1,_=env.p.getBasePositionAndOrientation(env.robot)
        delta=np.asarray(p1[:2],np.float32)-np.asarray(p0[:2],np.float32)
        return float(delta@heading)
    finally:
        env.close()

def _yaw_delta_for_turn_sign(turn_sign):
    CFG.TURN_SIGN=float(turn_sign)
    env=SingleRoomSocialEnv(
        seed=CFG.SEED+7102,
        scenario="empty",
        sensor_mode="command_only",
        enable_depth_shield=False)
    try:
        env.reset(seed=CFG.SEED+7102)
        _,q0=env.p.getBasePositionAndOrientation(env.robot)
        yaw0=env.p.getEulerFromQuaternion(q0)[2]

        for _ in range(14):
            _,_,te,tr,_=env.step(np.array([0.0,0.55],np.float32))
            if te or tr: break

        _,q1=env.p.getBasePositionAndOrientation(env.robot)
        yaw1=env.p.getEulerFromQuaternion(q1)[2]
        return float(_wrap_pi(yaw1-yaw0))
    finally:
        env.close()

# 6B) Measure forward and turning signs instead of assuming URDF conventions, then verify that the robot can physically reach a goal before PPO training.
def automatic_actuator_calibration():
    
    forward_tests={}
    for sign in (+1.0,-1.0):
        forward_tests[sign]=_forward_projection_for_sign(sign)

    best_drive=max(forward_tests,key=forward_tests.get)
    CFG.DRIVE_SIGN=float(best_drive)

    print("[calibration] forward projections:",
          ", ".join(f"sign={k:+.0f}:{v:+.3f}m" for k,v in forward_tests.items()))
    print(f"[calibration] selected DRIVE_SIGN={CFG.DRIVE_SIGN:+.0f}")

    if forward_tests[best_drive] <= 0.08:
        raise RuntimeError(
            "ACTUATOR CALIBRATION FAILED: neither wheel sign produced reliable "
            "forward motion. Check wheel axes/contact/friction.")

    
    turn_tests={}
    for ts in (+1.0,-1.0):
        turn_tests[ts]=_yaw_delta_for_turn_sign(ts)

    best_turn=max(turn_tests,key=turn_tests.get)
    CFG.TURN_SIGN=float(best_turn)

    print("[calibration] yaw deltas:",
          ", ".join(f"sign={k:+.0f}:{v:+.3f}rad" for k,v in turn_tests.items()))
    print(f"[calibration] selected TURN_SIGN={CFG.TURN_SIGN:+.0f}")

    if turn_tests[best_turn] <= 0.10:
        raise RuntimeError(
            "ACTUATOR CALIBRATION FAILED: neither turn sign produced reliable "
            "positive yaw.")

def deterministic_goal_teacher_preflight():
    
    
    env=SingleRoomSocialEnv(
        seed=CFG.SEED+7103,
        scenario="empty",
        sensor_mode="command_only",
        enable_depth_shield=False)
    try:
        obs,_=env.reset(seed=CFG.SEED+7103)
        final_info=None

        for _ in range(CFG.MAX_EPISODE_STEPS):
            cmd=obs["cmd"]
            goal_err=float(math.atan2(float(cmd[1]),float(cmd[2])))
            alignment=max(0.0,math.cos(goal_err))

            
            
            forward=float(
                CFG.GOAL_ASSIST_FORWARD_MIN +
                (CFG.GOAL_ASSIST_FORWARD_MAX-CFG.GOAL_ASSIST_FORWARD_MIN)
                * alignment**2)
            if abs(goal_err) > 0.75:
                forward=min(forward,0.25)

            turn=float(np.clip(
                CFG.GOAL_ASSIST_TURN_GAIN*goal_err/(math.pi/2),
                -1.0,1.0))

            obs,_,term,trunc,info=env.step(
                np.array([forward,turn],np.float32))

            if term or trunc:
                final_info=info
                break

        ep=(final_info or {}).get("episode",{})
        success=int(ep.get("success",0))
        reason=ep.get("terminal_reason","none")
        path=float(ep.get("path_length_m",float("nan")))
        print(
            f"[goal teacher test] success={success} "
            f"reason={reason} steps={env.t} path={path:.2f}m")

        if success != 1:
            raise RuntimeError(
                "GOAL-REACHABILITY PREFLIGHT FAILED after actuator calibration. "
                "Training aborted because PPO data would be invalid.")
    finally:
        env.close()

automatic_actuator_calibration()
deterministic_goal_teacher_preflight()
print("[05B/18] PASS — actuator calibration and goal reachability verified.")





stage(6, "Defining CNN, ViT, Cartesian-MDTA and Log-Polar MDTA encoders")

class ChannelLayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.bias=nn.Parameter(torch.zeros(dim)); self.eps=eps
    def forward(self,x):
        mean=x.mean(1,keepdim=True); var=((x-mean)**2).mean(1,keepdim=True)
        x=(x-mean)/torch.sqrt(var+self.eps)
        return x*self.weight[None,:,None,None]+self.bias[None,:,None,None]

class MDTA(nn.Module):
    def __init__(self,dim,heads=4):
        super().__init__(); assert dim%heads==0
        self.heads=heads; self.cph=dim//heads
        self.temperature=nn.Parameter(torch.ones(heads,1,1))
        self.qkv=nn.Conv2d(dim,dim*3,1,bias=False)
        self.qkv_dw=nn.Conv2d(dim*3,dim*3,3,padding=1,groups=dim*3,bias=False)
        self.out=nn.Conv2d(dim,dim,1,bias=False)
    def forward(self,x):
        B,C,H,W=x.shape
        q,k,v=self.qkv_dw(self.qkv(x)).chunk(3,dim=1)
        def r(t): return t.reshape(B,self.heads,self.cph,H*W)
        q,k,v=r(q),r(k),r(v)
        q,k=F.normalize(q,dim=-1),F.normalize(k,dim=-1)
        attn=F.softmax(torch.matmul(q,k.transpose(-2,-1))*self.temperature,dim=-1)
        z=torch.matmul(attn,v).reshape(B,C,H,W)
        return self.out(z)

class GDFN(nn.Module):
    def __init__(self,dim,expansion=2.2):
        super().__init__(); hidden=int(dim*expansion)
        self.inp=nn.Conv2d(dim,hidden*2,1,bias=False)
        self.dw=nn.Conv2d(hidden*2,hidden*2,3,padding=1,groups=hidden*2,bias=False)
        self.out=nn.Conv2d(hidden,dim,1,bias=False)
    def forward(self,x):
        a,b=self.dw(self.inp(x)).chunk(2,dim=1)
        return self.out(F.gelu(a)*b)

class MDTABlock(nn.Module):
    def __init__(self,dim,heads=4):
        super().__init__(); self.n1=ChannelLayerNorm(dim); self.a=MDTA(dim,heads); self.n2=ChannelLayerNorm(dim); self.f=GDFN(dim)
    def forward(self,x):
        x=x+self.a(self.n1(x)); x=x+self.f(self.n2(x)); return x

class LogPolar(nn.Module):
    def __init__(self,H=64,W=128,beta=3.0):
        super().__init__()
        u=torch.linspace(0,1,H); th=torch.linspace(-math.pi,math.pi,W)
        uu,tt=torch.meshgrid(u,th,indexing="ij")
        r=(torch.exp(beta*uu)-1)/(math.exp(beta)-1)
        grid=torch.stack([r*torch.cos(tt),r*torch.sin(tt)],dim=-1)
        self.register_buffer("grid",grid[None])
    def forward(self,x):
        return F.grid_sample(x,self.grid.expand(x.shape[0],-1,-1,-1),mode="bilinear",padding_mode="border",align_corners=True)

# 7) Encoder ablation: compare CNN, ViT, MDTA, and the proposed log-polar MDTA under the same downstream training protocol.
class CNNFrameEncoder(nn.Module):
    out_dim=192
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(4,32,5,2,2),nn.GELU(),
            nn.Conv2d(32,48,3,2,1),nn.GELU(),
            nn.Conv2d(48,64,3,2,1),nn.GELU(),
            nn.Conv2d(64,96,3,2,1),nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc=nn.Sequential(nn.Linear(96,192),nn.LayerNorm(192),nn.GELU())
    def forward(self,x): return self.fc(self.net(x).flatten(1))

class TinyViTFrameEncoder(nn.Module):
    out_dim=192
    def __init__(self,patch=8,dim=128,heads=4,depth=3):
        super().__init__()
        self.patch=nn.Conv2d(4,dim,kernel_size=patch,stride=patch)
        n=(CFG.CAM_H//patch)*(CFG.CAM_W//patch)
        self.pos=nn.Parameter(torch.zeros(1,n,dim)); nn.init.trunc_normal_(self.pos,std=.02)
        layer=nn.TransformerEncoderLayer(dim,heads,dim*3,dropout=.05,activation="gelu",batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(layer,depth)
        self.proj=nn.Sequential(nn.Linear(dim,192),nn.LayerNorm(192),nn.GELU())
    def forward(self,x):
        z=self.patch(x).flatten(2).transpose(1,2)
        z=self.tr(z+self.pos[:,:z.shape[1]])
        return self.proj(z.mean(1))

class MDTAFrameEncoder(nn.Module):
    out_dim=192
    def __init__(self,logpolar=False):
        super().__init__(); self.use_lp=logpolar
        self.lp=LogPolar(64,128,3.0) if logpolar else None
        branches=2 if logpolar else 1
        self.stems=nn.ModuleList([nn.Sequential(nn.Conv2d(4,48,3,2,1),nn.GELU(),nn.Conv2d(48,48,3,1,1)) for _ in range(branches)])
        self.fuse=nn.Conv2d(48*branches,48,1)
        self.s1=nn.Sequential(MDTABlock(48,4),MDTABlock(48,4),MDTABlock(48,4))
        self.down=nn.Sequential(nn.Conv2d(48,72,3,2,1),nn.GELU())
        self.s2=nn.Sequential(MDTABlock(72,4),MDTABlock(72,4))
        self.proj=nn.Sequential(nn.Linear(144,192),nn.LayerNorm(192),nn.GELU())
    def forward(self,x):
        raw=F.interpolate(x,size=(64,128),mode="bilinear",align_corners=False)
        feats=[self.stems[0](raw)]
        if self.use_lp:
            feats.append(self.stems[1](self.lp(x)))
        z=self.fuse(torch.cat(feats,dim=1)); z=self.s1(z); z=self.down(z); z=self.s2(z)
        avg=F.adaptive_avg_pool2d(z,1).flatten(1); mx=F.adaptive_max_pool2d(z,1).flatten(1)
        return self.proj(torch.cat([avg,mx],dim=1))

class SocialActorCritic(nn.Module):
    def __init__(self,encoder="cnn"):
        super().__init__()
        if encoder=="cnn": self.frame_encoder=CNNFrameEncoder()
        elif encoder=="vit": self.frame_encoder=TinyViTFrameEncoder()
        elif encoder=="mdta": self.frame_encoder=MDTAFrameEncoder(False)
        elif encoder=="lp_mdta": self.frame_encoder=MDTAFrameEncoder(True)
        else: raise ValueError(encoder)
        D=self.frame_encoder.out_dim
        self.temporal_pos=nn.Parameter(torch.zeros(1,CFG.HISTORY,D)); nn.init.trunc_normal_(self.temporal_pos,std=.02)
        layer=nn.TransformerEncoderLayer(D,4,384,dropout=.05,activation="gelu",batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2)
        self.predictor=nn.Sequential(nn.Linear(D,D),nn.GELU(),nn.Linear(D,D))
        self.fusion=nn.Sequential(nn.Linear(D+5,192),nn.LayerNorm(192),nn.GELU(),nn.Linear(192,128),nn.GELU())
        self.actor=nn.Linear(128,2); self.reward_value=nn.Linear(128,1); self.cost_value=nn.Linear(128,1)
        self.log_std=nn.Parameter(torch.tensor([-0.55,-0.45]))

    def encode(self,vision):
        B,C,H,W=vision.shape; T=CFG.HISTORY
        x=vision.reshape(B,T,4,H,W).reshape(B*T,4,H,W)
        f=self.frame_encoder(x); D=f.shape[-1]; f=f.reshape(B,T,D)
        z=self.temporal(f+self.temporal_pos[:,:T])
        return f,z

    def forward(self,vision,cmd):
        f,z=self.encode(vision); latest=z[:,-1]
        h=self.fusion(torch.cat([latest,cmd],dim=-1))
        mu=self.actor(h); rv=self.reward_value(h).squeeze(-1); cv=self.cost_value(h).squeeze(-1)
        if CFG.HISTORY>=2:
            pred=F.normalize(self.predictor(z[:,-2]),dim=-1); target=F.normalize(f[:,-1].detach(),dim=-1)
            aux=(1-(pred*target).sum(-1)).mean()
        else: aux=torch.zeros((),device=vision.device)
        return mu,rv,cv,aux

    def _dist(self,mu): return Normal(mu,self.log_std.exp().expand_as(mu))
    @staticmethod
    def squash_log_prob(dist,u,a):
        return dist.log_prob(u).sum(-1)-torch.log(1-a.pow(2)+1e-6).sum(-1)

    @torch.no_grad()
    def act(self,vision,cmd,deterministic=False):
        mu,rv,cv,_=self(vision,cmd); dist=self._dist(mu); u=mu if deterministic else dist.sample(); a=torch.tanh(u)
        logp=self.squash_log_prob(dist,u,a); return a,u,logp,rv,cv

    def evaluate_actions(self,vision,cmd,u):
        mu,rv,cv,aux=self(vision,cmd); dist=self._dist(mu); a=torch.tanh(u)
        logp=self.squash_log_prob(dist,u,a); entropy=dist.entropy().sum(-1)
        return logp,entropy,rv,cv,aux,mu




stage(7, "Running model forward/backward/NaN self-tests before long training")

def tensor_obs(obs):
    return (torch.from_numpy(obs["vision"][None]).float().to(DEVICE),
            torch.from_numpy(obs["cmd"][None]).float().to(DEVICE))

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

for enc in ["cnn","vit","mdta","lp_mdta"]:
    m=SocialActorCritic(enc).to(DEVICE)
    v,c=tensor_obs(obs)
    mu,rv,cv,aux=m(v,c)
    assert mu.shape==(1,2) and rv.shape==(1,) and cv.shape==(1,)
    loss=mu.pow(2).mean()+rv.pow(2).mean()+cv.pow(2).mean()+aux
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    print(f"[model test] {enc:8s}: params={count_params(m):,}, output={mu.detach().cpu().numpy().round(3)}")
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print("[model self-test] PASS")






stage(8, "Defining fast reactive + privileged social-DWA baselines")

def tensor_obs(obs):
    return (torch.from_numpy(obs["vision"]).unsqueeze(0).float().to(DEVICE),
            torch.from_numpy(obs["cmd"]).unsqueeze(0).float().to(DEVICE))

def count_params(m): return sum(p.numel() for p in m.parameters())

# 9) Reference controllers separate basic goal following, depth-reactive behavior, and privileged DWA planning from learned policies.
class GoalSeekingPolicy:
    name="Goal-Seeking"
    def act_np(self,obs,env):
        
        
        cmd=obs["cmd"]
        turn=float(np.clip(np.arctan2(cmd[1],cmd[2])/math.pi,-1,1))
        forward=float(np.clip(0.72-0.28*abs(turn),0.30,0.72))
        return np.array([forward,turn],np.float32)

class DepthReactivePolicy:
    name="Depth-Reactive"
    def act_np(self,obs,env):
        depth=obs["vision"][-1]; H,W=depth.shape
        band=depth[int(.42*H):int(.90*H)]
        left=float(np.quantile(band[:,:W//3],.88))
        center=float(np.quantile(band[:,W//3:2*W//3],.90))
        right=float(np.quantile(band[:,2*W//3:],.88))
        cmd=obs["cmd"]
        goal_turn=np.arctan2(cmd[1],cmd[2])/math.pi
        hazard=max(left,center,right)
        forward=np.clip(1-1.30*hazard,0,1)
        avoid=(right-left)*1.45
        turn=np.clip(.85*goal_turn+avoid,-1,1)
        if center>.72: forward*=.15
        return np.array([forward,turn],np.float32)

class PrivilegedSocialDWA:
    name="Privileged-Social-DWA"
    def __init__(self):
        
        
        vv,ww=np.meshgrid(np.linspace(0.0,1.0,4,dtype=np.float32),
                          np.linspace(-1.0,1.0,7,dtype=np.float32),indexing="ij")
        self.vn=vv.reshape(-1); self.wn=ww.reshape(-1)
        self.v=self.vn*CFG.MAX_LINEAR; self.w=self.wn*CFG.MAX_ANGULAR
        self.dt=.25; self.horizon_steps=3

    @staticmethod
    def _static_clearance_batch(x,y,rects):
        best=np.full_like(x,999.,dtype=np.float32)
        for rx,ry,rw,rh in rects:
            cx=np.clip(x,rx,rx+rw); cy=np.clip(y,ry,ry+rh)
            d=np.hypot(x-cx,y-cy)
            inside=(x>=rx)&(x<=rx+rw)&(y>=ry)&(y<=ry+rh)
            if np.any(inside):
                ins=np.minimum.reduce([x-rx,rx+rw-x,y-ry,ry+rh-y])
                d=np.where(inside,-ins,d)
            best=np.minimum(best,d.astype(np.float32))
        return best

    def act_np(self,obs,env):
        rpos,yaw,goal,humans=env.privileged_state()
        n=len(self.v)
        x=np.full(n,float(rpos[0]),np.float32)
        y=np.full(n,float(rpos[1]),np.float32)
        th=np.full(n,float(yaw),np.float32)
        min_h=np.full(n,99.,np.float32)
        collision=np.zeros(n,dtype=bool)
        if humans:
            hp=np.stack([h[0] for h in humans],0).astype(np.float32)
            hv=np.stack([h[1] for h in humans],0).astype(np.float32)
        else:
            hp=np.empty((0,2),np.float32); hv=np.empty((0,2),np.float32)
        rects=env.wall_rects+env.furniture_rects

        for k in range(self.horizon_steps):
            th=th+self.w*self.dt
            x=x+self.v*np.cos(th)*self.dt
            y=y+self.v*np.sin(th)*self.dt
            tt=(k+1)*self.dt
            if len(hp):
                pred=hp[None,:,:]+hv[None,:,:]*tt
                cand=np.stack([x,y],1)[:,None,:]
                clear=np.linalg.norm(cand-pred,axis=2)-(CFG.ROBOT_RADIUS+CFG.HUMAN_RADIUS)
                step_min=clear.min(axis=1)
                min_h=np.minimum(min_h,step_min.astype(np.float32))
                collision |= step_min<.04
            collision |= self._static_clearance_batch(x,y,rects)<CFG.ROBOT_RADIUS

        gd=np.hypot(x-goal[0],y-goal[1])
        desired=np.arctan2(goal[1]-y,goal[0]-x)
        heading=np.abs((desired-th+np.pi)%(2*np.pi)-np.pi)
        score=-1.8*gd-.20*heading+.45*self.v+.65*np.minimum(min_h,1.3)
        score -= 2.5*np.maximum(0.,CFG.PERSONAL_CLEARANCE-min_h)
        score=np.where(collision,score-100.,score)
        i=int(np.argmax(score))
        return np.array([self.vn[i],self.wn[i]],np.float32)




def training_exploration_override(action, env, global_step, max_steps, rng):
    """Return (executed_action, assisted, hazard, alpha, goal_err).

    The teacher is deliberately weak and non-privileged: it uses only the same
    egocentric goal command already supplied to the learned policy. It cannot
    inspect simulator human positions. MiDaS gates forward assistance. Assisted
    transitions are masked out of PPO's actor likelihood-ratio objective.
    Evaluation never calls this function.
    """
    a=np.clip(np.asarray(action,np.float32).copy(),-1,1)
    if (not CFG.GOAL_ASSIST) or max_steps<=0:
        return a,False,0.0,0.0,0.0

    frac=float(global_step)/float(max_steps)
    if frac>=CFG.GOAL_ASSIST_END_FRAC:
        return a,False,0.0,0.0,0.0

    
    cmd=env._goal_command()
    goal_err=float(math.atan2(float(cmd[1]),float(cmd[2])))
    alignment=max(0.0, math.cos(goal_err))

    
    hazard=0.0
    if env.depth_cache is not None:
        H,W=env.depth_cache.shape
        crop=env.depth_cache[int(.38*H):int(.92*H),int(.34*W):int(.66*W)]
        if crop.size:
            hazard=float(np.quantile(crop,.90))

    q=np.clip(frac/max(CFG.GOAL_ASSIST_END_FRAC,1e-6),0,1)
    alpha=(1-q)*CFG.GOAL_ASSIST_ALPHA_START + q*CFG.GOAL_ASSIST_ALPHA_END

    
    teacher_turn=float(np.clip(CFG.GOAL_ASSIST_TURN_GAIN*goal_err/(math.pi/2),-1,1))
    teacher_forward=float(CFG.GOAL_ASSIST_FORWARD_MIN +
                          (CFG.GOAL_ASSIST_FORWARD_MAX-CFG.GOAL_ASSIST_FORWARD_MIN)*alignment)

    target=a.copy()
    
    
    target[1]=(1-alpha)*a[1] + alpha*teacher_turn
    if hazard < CFG.GOAL_ASSIST_HAZARD_LIMIT:
        target[0]=(1-alpha)*a[0] + alpha*teacher_forward
        
        floor=CFG.GOAL_ASSIST_FORWARD_MIN*(0.45+0.55*alignment)*(1-q)
        target[0]=max(float(target[0]),float(floor))
    else:
        
        
        target[0]=min(float(target[0]),float(a[0]))

    target=np.clip(target,-1,1).astype(np.float32)
    assisted=bool(np.linalg.norm(target-a) > CFG.GOAL_ASSIST_MIN_CHANGE)
    return target,assisted,float(hazard),float(alpha),float(goal_err)




# 10) Shared PPO/RCPPO training logic uses the same optimization rules for every learned method; risk-constrained variants additionally learn a Lagrange multiplier.
stage(9, "Defining adaptive-budget PPO/RCPPO trainer")

def gae(rewards,values,dones,last_value,gamma=CFG.GAMMA,lam=CFG.GAE_LAMBDA):
    n=len(rewards); adv=np.zeros(n,np.float32); last=0.
    for t in reversed(range(n)):
        nv=last_value if t==n-1 else values[t+1]
        nonterm=1-dones[t]
        delta=rewards[t]+gamma*nv*nonterm-values[t]
        last=delta+gamma*lam*nonterm*last
        adv[t]=last
    return adv,adv+values

def train_policy(model,tag,use_constraint,seed_offset=0):
    env=SingleRoomSocialEnv(seed=CFG.SEED+seed_offset,scenario="train")
    opt=torch.optim.AdamW(model.parameters(),lr=CFG.LR,weight_decay=1e-4)
    obs,_=env.reset(seed=CFG.SEED+seed_offset)
    steps=0; update=0
    lagrange=.5 if use_constraint else 0.
    rng=np.random.default_rng(CFG.SEED+seed_offset+991)
    train_rows=[]; episode_rows=[]; recent=[]; ep_ret=0.
    total_forced=0
    start=time.time(); stop_reason="max_budget"
    pbar=tqdm(total=CFG.MAX_TRAIN_STEPS,desc=f"TRAIN {tag}",unit="step",dynamic_ncols=True)

    while steps<CFG.MAX_TRAIN_STEPS:
        nroll=min(CFG.ROLLOUT,CFG.MAX_TRAIN_STEPS-steps)
        V,C,U,EA,LP,RW,CO,DN,RV,CV,PM=[],[],[],[],[],[],[],[],[],[],[]

        for _ in range(nroll):
            v,c=tensor_obs(obs)
            with torch.inference_mode():
                action,u,logp,rv,cv=model.act(v,c,False)
            raw_action=action[0].cpu().numpy()
            exec_action,forced,force_hazard,assist_alpha,goal_err=training_exploration_override(
                raw_action,env,steps,CFG.MAX_TRAIN_STEPS,rng)
            total_forced += int(forced)
            env.forced_exploration_total += int(forced)
            nxt,reward,term,trunc,info=env.step(exec_action)
            info["forced_exploration"]=float(forced)
            info["force_hazard"]=float(force_hazard)
            info["assist_alpha"]=float(assist_alpha)
            info["assist_goal_error_rad"]=float(goal_err)
            done=term or trunc
            V.append(obs["vision"].astype(np.float16)); C.append(obs["cmd"])
            U.append(u[0].cpu().numpy()); EA.append(exec_action.copy()); LP.append(float(logp.item()))
            RW.append(float(reward)); CO.append(float(info["safety_cost"]))
            DN.append(float(done)); RV.append(float(rv.item())); CV.append(float(cv.item()))
            
            
            PM.append(0.0 if forced else 1.0)
            ep_ret += reward; steps += 1

            if done:
                row={"step":steps,"return":ep_ret,**info["episode"],
                     "forced_exploration_fraction":float(env.forced_exploration_total/max(env.t,1))}
                episode_rows.append(row); recent.append(row)
                recent=recent[-CFG.EARLY_STOP_WINDOW:]
                obs,_=env.reset(); ep_ret=0.
            else:
                obs=nxt

        v,c=tensor_obs(obs)
        with torch.inference_mode():
            _,last_rv,last_cv,_=model(v,c)

        rw=np.asarray(RW,np.float32); co=np.asarray(CO,np.float32)
        dn=np.asarray(DN,np.float32); rv=np.asarray(RV,np.float32); cv=np.asarray(CV,np.float32)
        ar,rr=gae(rw,rv,dn,float(last_rv.item()))
        ac,rc=gae(co,cv,dn,float(last_cv.item()))
        ar=(ar-ar.mean())/(ar.std()+1e-8)
        ac=(ac-ac.mean())/(ac.std()+1e-8)

        vt=torch.from_numpy(np.asarray(V)).float().to(DEVICE)
        ct=torch.from_numpy(np.asarray(C)).float().to(DEVICE)
        ut=torch.from_numpy(np.asarray(U)).float().to(DEVICE)
        eat=torch.from_numpy(np.asarray(EA)).float().to(DEVICE)
        oldlp=torch.tensor(LP,dtype=torch.float32,device=DEVICE)
        art=torch.from_numpy(ar).float().to(DEVICE)
        act=torch.from_numpy(ac).float().to(DEVICE)
        rrt=torch.from_numpy(rr).float().to(DEVICE)
        rct=torch.from_numpy(rc).float().to(DEVICE)
        pmt=torch.tensor(PM,dtype=torch.float32,device=DEVICE)

        idx=np.arange(nroll); pl=[]; auxl=[]; bcl=[]
        for _epoch in range(CFG.EPOCHS):
            np.random.shuffle(idx)
            for s in range(0,nroll,CFG.MINIBATCH):
                mb=idx[s:s+CFG.MINIBATCH]
                logp,entropy,rvp,cvp,aux,mu=model.evaluate_actions(vt[mb],ct[mb],ut[mb])
                ratio=torch.exp(logp-oldlp[mb])
                reward_s=torch.min(ratio*art[mb],
                                   torch.clamp(ratio,1-CFG.CLIP,1+CFG.CLIP)*art[mb])
                cost_s=ratio*act[mb]
                mask=pmt[mb]
                
                
                
                actor_obj=(reward_s-lagrange*cost_s)*mask
                pol=-actor_obj.sum()/mask.sum().clamp_min(1.0)
                
                
                
                
                assist_mask=(1.0-mask)
                pred_action=torch.tanh(mu)
                per_bc=(pred_action-eat[mb]).pow(2).mean(dim=-1)
                bc=(per_bc*assist_mask).sum()/assist_mask.sum().clamp_min(1.0)
                loss=(pol + CFG.ASSIST_BC_COEF*bc + CFG.VALUE_COEF*F.mse_loss(rvp,rrt[mb])
                      + CFG.COST_VALUE_COEF*F.mse_loss(cvp,rct[mb])
                      - CFG.ENTROPY_COEF*entropy.mean() + CFG.AUX_COEF*aux)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),CFG.MAX_GRAD_NORM)
                opt.step()
                pl.append(float(pol.item())); auxl.append(float(aux.item())); bcl.append(float(bc.item()))

        mean_cost=float(co.mean())
        if use_constraint:
            lagrange=float(np.clip(lagrange+CFG.DUAL_LR*(mean_cost-CFG.COST_LIMIT),
                                   0,CFG.MAX_LAGRANGE))

        update+=1
        succ=float(np.mean([x["success"] for x in recent])) if recent else np.nan
        hcol=float(np.mean([x["human_collision"] for x in recent])) if recent else np.nan
        timeout=float(np.mean([x["timeout"] for x in recent])) if recent else np.nan
        elapsed=time.time()-start
        train_rows.append({
            "update":update,"steps":steps,
            "mean_rollout_reward":float(rw.mean()),
            "mean_safety_cost":mean_cost,"lagrange":lagrange,
            "policy_loss":float(np.mean(pl)),"aux_loss":float(np.mean(auxl)),
            "assist_bc_loss":float(np.mean(bcl)),
            "recent_success":succ,"recent_human_collision":hcol,
            "recent_timeout":timeout,
            "forced_exploration_fraction":total_forced/max(steps,1),
            "assisted_exploration_fraction":total_forced/max(steps,1),
            "unforced_actor_fraction":float(np.mean(PM)),
            "steps_per_second":steps/max(elapsed,1e-6)})

        pbar.update(nroll)
        pbar.set_postfix(R=f"{rw.mean():+.2f}",C=f"{mean_cost:.3f}",
                         lam=f"{lagrange:.2f}",
                         succ=f"{succ:.2f}" if np.isfinite(succ) else "--",
                         hcol=f"{hcol:.2f}" if np.isfinite(hcol) else "--",
                         assist=f"{total_forced/max(steps,1):.2f}",
                         case=env.scenario,midas=depth_model.calls)

        if (steps>=CFG.MIN_TRAIN_STEPS and len(recent)>=CFG.EARLY_STOP_WINDOW
            and succ>=CFG.EARLY_STOP_SUCCESS
            and hcol<=CFG.EARLY_STOP_HUMAN_COLLISION):
            stop_reason=f"early_stop_success_{succ:.2f}_hcol_{hcol:.2f}"
            break

    pbar.close(); env.close()
    print(f"[{tag}] stop={stop_reason} | steps={steps:,} | time={(time.time()-start)/60:.1f} min")
    return pd.DataFrame(train_rows),pd.DataFrame(episode_rows),steps,stop_reason




stage(10, "Training compact comparison suite in the same one-room curriculum")

# 11) Train all learned methods with the same curriculum, optimizer settings, maximum step budget, and early-stop rule.
METHODS={}; TRAIN_LOGS={}; EP_LOGS={}; TRAIN_META=[]
specs=[]
if CFG.TRAIN_CNN: specs.append(("CNN-PPO","cnn",False))
if CFG.TRAIN_VIT: specs.append(("ViT-PPO","vit",False))
if CFG.TRAIN_MDTA: specs.append(("MDTA-PPO","mdta",False))
if CFG.TRAIN_LP_MDTA_PPO: specs.append(("LP-MDTA-PPO","lp_mdta",False))
if CFG.TRAIN_PROPOSED: specs.append(("LP-MDTA-RCPPO","lp_mdta",True))

for i,(name,enc,constraint) in enumerate(specs):
    print("\n"+"#"*100)
    print(f"TRAINING {i+1}/{len(specs)}: {name} | {enc} | constrained={constraint}")
    print("#"*100,flush=True)
    torch.manual_seed(CFG.SEED+i)
    model=SocialActorCritic(enc).to(DEVICE)
    print(f"Parameters: {count_params(model):,}")
    tr,ep,used,reason=train_policy(model,name,constraint,seed_offset=i*100)
    METHODS[name]=model.eval(); TRAIN_LOGS[name]=tr; EP_LOGS[name]=ep
    TRAIN_META.append({"method":name,"encoder":enc,"steps_used":used,
                       "stop_reason":reason,"parameters":count_params(model),
                       "final_forced_exploration_fraction":float(tr.forced_exploration_fraction.iloc[-1]) if len(tr) else np.nan})
    tr.to_csv(ROOT/f"train_{name.replace('-','_')}.csv",index=False)
    ep.to_csv(ROOT/f"episodes_{name.replace('-','_')}.csv",index=False)
    torch.save({"method":name,"encoder":enc,"constrained":constraint,
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},
               ROOT/f"{name.replace('-','_')}.pt")

pd.DataFrame(TRAIN_META).to_csv(ROOT/"training_budget_used.csv",index=False)
METHODS["Goal-Seeking"]=GoalSeekingPolicy()
METHODS["Depth-Reactive"]=DepthReactivePolicy()
METHODS["Privileged-Social-DWA"]=PrivilegedSocialDWA()
assert len(METHODS)==8, f"Expected exactly 8 comparison methods, got {list(METHODS)}"
print("Comparison methods (8):", ", ".join(METHODS.keys()))




stage(11, "Evaluating named one-room cases on identical held-out seeds")

def act_method(method,obs,env):
    if isinstance(method,SocialActorCritic):
        v,c=tensor_obs(obs); t=time.perf_counter()
        with torch.inference_mode():
            a,_,_,_,_=method.act(v,c,True)
        return a[0].cpu().numpy(),(time.perf_counter()-t)*1000
    t=time.perf_counter()
    a=method.act_np(obs,env)
    return a,(time.perf_counter()-t)*1000

def _evaluation_sensor_profile(method):
    if isinstance(method,SocialActorCritic):
        return "full", True
    if isinstance(method,DepthReactivePolicy):
        return "full", True
    if isinstance(method,GoalSeekingPolicy):
        return "command_only", False
    if isinstance(method,PrivilegedSocialDWA):
        return "privileged", False
    return "full", True

# 12) Evaluate without training assistance on identical named scenarios and held-out seeds for a fair same-simulator comparison.
def evaluate_method(name,method,case,episodes,save_video=False):
    rows=[]; writer=None; vid=None; demo=None
    if save_video:
        vid=str(ROOT/f"demo_{name.replace('-','_')}_{case}.mp4")
        writer=imageio.get_writer(vid,fps=CFG.VIDEO_FPS,codec="libx264",quality=7)

    sensor_mode,depth_shield=_evaluation_sensor_profile(method)
    
    
    
    env=SingleRoomSocialEnv(seed=20_000,scenario=case,
                            sensor_mode=sensor_mode,
                            enable_depth_shield=depth_shield)
    try:
        for ep in tqdm(range(episodes),desc=f"EVAL {name} [{case}]",
                       unit="ep",dynamic_ncols=True):
            seed=20_000+ep
            obs,_=env.reset(seed=seed)
            done=False; ret=0.; inf=[]
            while not done:
                a,ms=act_method(method,obs,env); inf.append(ms)
                obs,r,te,tr,info=env.step(a); ret+=r; done=te or tr
                if writer is not None and ep==0 and env.t%2==0:
                    writer.append_data(env.render_overhead())

            rows.append({"method":name,"case":case,"episode":ep,
                         "return":ret,"inference_ms_mean":float(np.mean(inf)),
                         "forced_exploration_fraction":0.0,
                         **info["episode"]})
            if ep==0:
                demo=(np.asarray(env.trajectory),
                      [np.asarray(x) for x in env.human_trajectories],
                      list(env.wall_rects),list(env.furniture_rects),env.goal_xy.copy())
    finally:
        env.close()
        if writer is not None: writer.close()
    return pd.DataFrame(rows),vid,demo

ALL=[]; DEMOS_8={}
VIDEO_COMPARE_CASE="dense_5"
for case in TEST_CASES:
    for name,method in METHODS.items():
        
        
        
        df,vid,demo=evaluate_method(name,method,case,CFG.EVAL_EPISODES_PER_CASE,False)
        ALL.append(df)
        if case==VIDEO_COMPARE_CASE:
            DEMOS_8[name]=demo

EVAL=pd.concat(ALL,ignore_index=True)
EVAL.to_csv(ROOT/"all_episode_evaluations.csv",index=False)




stage(12, "Computing 95% CIs, per-case and aggregate comparisons")

def wilson_ci(k,n,alpha=.05):
    if n==0: return np.nan,np.nan
    z=stats.norm.ppf(1-alpha/2); p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

# 13) Report confidence intervals and paired bootstrap differences so the comparison includes uncertainty rather than only mean scores.
def mean_ci(x,alpha=.05):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return np.nan,np.nan,np.nan
    m=float(x.mean())
    if len(x)==1: return m,m,m
    h=float(stats.t.ppf(1-alpha/2,len(x)-1)*stats.sem(x))
    return m,m-h,m+h

def summarize(df):
    n=len(df); out={"episodes":n}
    for col in ["success","human_collision","static_collision","timeout"]:
        k=int(df[col].sum()); lo,hi=wilson_ci(k,n)
        out[col]=k/n; out[col+"_ci_lo"]=lo; out[col+"_ci_hi"]=hi
    for col in ["spl","path_length_m","min_clearance_m","discomfort_fraction",
                "ttc_risk_fraction","mean_social_cost","mean_ttc_cost",
                "smoothness","shield_fraction","max_human_penetration_m",
                "time_s","mean_speed_proxy","return","inference_ms_mean"]:
        m,lo,hi=mean_ci(df[col]); out[col]=m
        out[col+"_ci_lo"]=lo; out[col+"_ci_hi"]=hi
    return pd.Series(out)

SUMMARY=EVAL.groupby(["method","case"],sort=False).apply(summarize).reset_index()
OVERALL=EVAL.groupby("method",sort=False).apply(summarize).reset_index()
SUMMARY.to_csv(ROOT/"benchmark_by_test_case_95CI.csv",index=False)
OVERALL.to_csv(ROOT/"benchmark_overall_95CI.csv",index=False)

cols=["method","success","human_collision","static_collision","timeout","spl",
      "min_clearance_m","discomfort_fraction","ttc_risk_fraction",
      "mean_speed_proxy","inference_ms_mean"]
print("\nOVERALL CONTROLLED SINGLE-ROOM BENCHMARK\n")
print(OVERALL[cols].sort_values(["success","human_collision","spl"],
      ascending=[False,True,False]).to_string(index=False,
      float_format=lambda x:f"{x:.4f}"))

print("\nPER-TEST-CASE SUCCESS\n")
print(SUMMARY.pivot(index="method",columns="case",values="success")
      .to_string(float_format=lambda x:f"{x:.2f}"))

def paired_boot(prop,other,metric,nboot=2500,seed=123):
    keys=["case","episode"]
    a=prop[keys+[metric]].rename(columns={metric:"a"})
    b=other[keys+[metric]].rename(columns={metric:"b"})
    merged=a.merge(b,on=keys)
    d=(merged.a-merged.b).to_numpy(float)
    if len(d)==0: return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed)
    means=np.array([d[rng.integers(0,len(d),len(d))].mean()
                    for _ in range(nboot)])
    return float(d.mean()),float(np.percentile(means,2.5)),float(np.percentile(means,97.5))

PAIR=[]
if "LP-MDTA-RCPPO" in METHODS:
    prop=EVAL[EVAL.method=="LP-MDTA-RCPPO"]
    for other in METHODS:
        if other=="LP-MDTA-RCPPO": continue
        odf=EVAL[EVAL.method==other]
        for metric in ["success","human_collision","spl","discomfort_fraction",
                       "min_clearance_m","return"]:
            d,lo,hi=paired_boot(prop,odf,metric)
            PAIR.append({"comparison":other,"metric":metric,
                         "delta_proposed_minus_other":d,
                         "ci95_lo":lo,"ci95_hi":hi})
PAIR=pd.DataFrame(PAIR)
PAIR.to_csv(ROOT/"paired_bootstrap_vs_proposed.csv",index=False)

physics_max=float(EVAL.max_human_penetration_m.max())
physics_pass=physics_max<=.02
print(f"\nPhysics audit: max human-human penetration={physics_max:.4f} m -> {'PASS' if physics_pass else 'FAIL'}")




stage(13, "Generating compact benchmark plots")

plt.figure(figsize=(10,5))
for name,df in TRAIN_LOGS.items():
    plt.plot(df.steps,df.mean_rollout_reward,label=name)
plt.xlabel("Environment steps"); plt.ylabel("Mean rollout reward")
plt.title("Single-room training reward"); plt.grid(alpha=.25); plt.legend()
plt.tight_layout(); plt.savefig(ROOT/"training_reward.png",dpi=180); plt.show()

for metric,title in [
    ("success","Overall success"),
    ("human_collision","Overall human collision"),
    ("spl","Overall SPL"),
    ("discomfort_fraction","Overall personal-space violation")]:
    d=OVERALL.copy()
    plt.figure(figsize=(10,4.5))
    plt.bar(d.method,d[metric]); plt.xticks(rotation=25,ha="right")
    plt.ylabel(metric); plt.title(title); plt.tight_layout()
    plt.savefig(ROOT/f"overall_{metric}.png",dpi=180); plt.show()

pivot=SUMMARY.pivot(index="method",columns="case",values="success")
plt.figure(figsize=(10,5))
plt.imshow(pivot.values,aspect="auto",vmin=0,vmax=1)
plt.colorbar(label="Success rate")
plt.xticks(range(len(pivot.columns)),pivot.columns,rotation=20)
plt.yticks(range(len(pivot.index)),pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        plt.text(j,i,f"{pivot.values[i,j]:.2f}",ha="center",va="center")
plt.title("Success by controlled test case"); plt.tight_layout()
plt.savefig(ROOT/"success_case_matrix.png",dpi=180); plt.show()





stage(14, "Writing evaluation report")

case_md=SUMMARY[["method","case","success","human_collision","spl",
                 "min_clearance_m","discomfort_fraction",
                 "mean_speed_proxy","inference_ms_mean"]].to_markdown(
                     index=False,floatfmt=".4f")
overall_md=OVERALL[cols].to_markdown(index=False,floatfmt=".4f")
train_md=pd.DataFrame(TRAIN_META).to_markdown(index=False)
pair_md=PAIR.to_markdown(index=False,floatfmt=".4f") if len(PAIR) else "No pairwise table."
lit_md=LITERATURE_CONTEXT.to_markdown(index=False)

# 15) Generate a reproducible Markdown report from the actual run outputs, including protocol, results, physics checks, and interpretation limits.
REPORT = f"""# Physics-Enabled Spacious Single-Room LP-MDTA-RCPPO Benchmark

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Why this run exists
This experiment intentionally replaces the expensive 150k-step multi-room run with a controlled spacious-room benchmark. All learned methods share the same curriculum, maximum step budget, early-stop rule, and training-only exploration intervention schedule. Evaluation uses named held-out cases with identical seeds and contains zero forced exploration.

## Exploration reliability protocol
During early training, a MiDaS-gated minimum forward-motion intervention prevents trivial stationary data collection when the visible near field is clear. Every overridden transition is logged. Forced transitions are excluded from PPO's actor likelihood-ratio objective but remain available to reward/cost critics as physically realized transitions. The intervention decays completely to zero before training ends. All reported evaluation episodes are fully unassisted.

## Compute
- Device: {DEVICE}
- Max training steps/method: {CFG.MAX_TRAIN_STEPS:,}
- Minimum steps before early stop: {CFG.MIN_TRAIN_STEPS:,}
- Evaluation episodes/case: {CFG.EVAL_EPISODES_PER_CASE}
- RGB: {CFG.CAM_W}x{CFG.CAM_H}; history={CFG.HISTORY}; MiDaS stride={CFG.MIDAS_STRIDE}

## Training budgets actually used
{train_md}

## Overall same-simulator comparison
{overall_md}

## Named test cases
{case_md}

## Paired bootstrap: proposed minus comparator
For success/SPL/clearance positive is favorable. For collision/discomfort negative is favorable.

{pair_md}

## Physics audit
Maximum observed human-human penetration = {physics_max:.4f} m. Audit: {'PASS' if physics_pass else 'FAIL'}.

## External literature context — not a direct leaderboard
{lit_md}

Published numbers above use different simulators, sensors and protocols. They are contextual only. The scientifically valid direct comparison in this run is the same-room, same-seed table above.

## Interpretation rule
A low-collision method with near-zero success and near-zero speed is a freeze policy, not successful social navigation. Always interpret success, SPL, timeout, mean speed, collision and discomfort together.
"""
(ROOT/"EVALUATION_REPORT.md").write_text(REPORT)
print(REPORT[:9000])





stage(14, "Rendering physics-result 2x4 Matplotlib comparison video for all 8 methods")

def _rect_outline(ax, rect, lw=1.2):
    x,y,w,h=rect
    ax.add_patch(patches.Rectangle((x,y),w,h,fill=False,linewidth=lw))

def render_eight_method_video(demos, case, out_path):
    names=list(METHODS.keys())
    assert len(names)==8 and all(n in demos for n in names)
    max_frames=max(len(demos[n][0]) for n in names)
    nframes=min(max_frames,240)
    frame_ids=np.linspace(0,max_frames-1,nframes).astype(int)
    writer=imageio.get_writer(str(out_path),fps=CFG.VIDEO_FPS,codec="libx264",quality=7)

    
    
    fig,axes=plt.subplots(2,4,figsize=(16,10),dpi=100)
    fig.suptitle(f"Physics-enabled social navigation — same held-out seed — case: {case}\n"
                 "Green=robot | Red=humans | Gold=goal | line=robot trajectory",fontsize=14)
    artists={}
    for ax,name in zip(axes.ravel(),names):
        traj,human_trajs,walls,furn,goal_xy=demos[name]
        for rect in walls:
            x,y,w,h=rect; ax.add_patch(patches.Rectangle((x,y),w,h,alpha=.20))
        for rect in furn:
            x,y,w,h=rect; ax.add_patch(patches.Rectangle((x,y),w,h,alpha=.35))
        line,=ax.plot([],[],linewidth=1.5)
        robot=patches.Circle((0,0),CFG.ROBOT_RADIUS,fill=True,alpha=.85)
        ax.add_patch(robot)
        humans=[]
        for _ in human_trajs:
            c=patches.Circle((20,20),CFG.HUMAN_RADIUS,fill=True,alpha=.75)
            ax.add_patch(c); humans.append(c)
        ax.scatter([goal_xy[0]],[goal_xy[1]],marker="*",s=150)
        rr=EVAL[(EVAL.method==name)&(EVAL.case==case)&(EVAL.episode==0)].iloc[0]
        status=(f"S={int(rr.success)} HC={int(rr.human_collision)} SC={int(rr.static_collision)} "
                f"SPL={rr.spl:.2f}\nclear={rr.min_clearance_m:.2f}m v={rr.mean_speed_proxy:.2f}m/s")
        ax.set_title(name,fontsize=10)
        ax.text(.02,.98,status,transform=ax.transAxes,va="top",ha="left",fontsize=7,
                bbox=dict(boxstyle="round",alpha=.65))
        ax.set_xlim(-CFG.ROOM_X/2-.3,CFG.ROOM_X/2+.3); ax.set_ylim(-CFG.ROOM_Y/2-.3,CFG.ROOM_Y/2+.3)
        ax.set_aspect("equal"); ax.grid(alpha=.18); ax.set_xticks([]); ax.set_yticks([])
        artists[name]=(line,robot,humans)
    fig.tight_layout(rect=[0,0,1,.94])

    for global_idx in tqdm(frame_ids,desc="VIDEO 8-method 2x4",unit="frame",dynamic_ncols=True):
        for name in names:
            traj,human_trajs,_,_,_=demos[name]
            line,robot,hcircles=artists[name]
            ridx=min(global_idx,max(len(traj)-1,0))
            if len(traj):
                t=np.asarray(traj); line.set_data(t[:ridx+1,0],t[:ridx+1,1]); robot.center=tuple(t[ridx])
            for c,ht in zip(hcircles,human_trajs):
                if len(ht):
                    hi=min(global_idx,len(ht)-1); c.center=tuple(np.asarray(ht[hi]))
                else:
                    c.center=(20,20)
        fig.canvas.draw()
        rgba=np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(rgba[:,:,:3].copy())
    writer.close(); plt.close(fig)
    return str(out_path)

COMPARISON_VIDEO=render_eight_method_video(
    DEMOS_8,VIDEO_COMPARE_CASE,ROOT/f"comparison_8_algorithms_{VIDEO_COMPARE_CASE}.mp4")
print("8-method comparison video:",COMPARISON_VIDEO)


fig,axes=plt.subplots(2,4,figsize=(16,10),dpi=140)
for ax,name in zip(axes.ravel(),METHODS.keys()):
    traj,human_trajs,walls,furn,goal_xy=DEMOS_8[name]
    for rect in walls:
        x,y,w,h=rect; ax.add_patch(patches.Rectangle((x,y),w,h,alpha=.20))
    for rect in furn:
        x,y,w,h=rect; ax.add_patch(patches.Rectangle((x,y),w,h,alpha=.35))
    if len(traj):
        t=np.asarray(traj); ax.plot(t[:,0],t[:,1],linewidth=1.8)
        ax.scatter([t[0,0]],[t[0,1]],marker="o",s=40)
        ax.scatter([t[-1,0]],[t[-1,1]],marker="x",s=55)
    ax.scatter([goal_xy[0]],[goal_xy[1]],marker="*",s=120)
    ax.set_title(name); ax.set_aspect("equal")
    ax.set_xlim(-CFG.ROOM_X/2-.3,CFG.ROOM_X/2+.3)
    ax.set_ylim(-CFG.ROOM_Y/2-.3,CFG.ROOM_Y/2+.3)
    ax.grid(alpha=.18)
fig.suptitle(f"Eight algorithms — held-out {VIDEO_COMPARE_CASE} trajectories",fontsize=14)
fig.tight_layout(rect=[0,0,1,.96])
fig.savefig(ROOT/f"comparison_8_algorithms_{VIDEO_COMPARE_CASE}_trajectories.png",bbox_inches="tight")
plt.close(fig)




# 16) Save the proposed policy and deployment metadata separately from the full research artifact bundle.
stage(15, "Saving proposed model + deployment metadata")

if "LP-MDTA-RCPPO" in METHODS:
    proposed=METHODS["LP-MDTA-RCPPO"]
    metadata={
        "algorithm":"LP-MDTA-RCPPO",
        "experiment":"controlled_single_room_fast",
        "camera_height_m":CFG.CAMERA_HEIGHT_M,
        "vision_shape":[CFG.HISTORY*4,CFG.CAM_H,CFG.CAM_W],
        "cmd_dim":5,"action_dim":2,
        "max_linear_mps":CFG.MAX_LINEAR,
        "max_angular_radps":CFG.MAX_ANGULAR,
        "midas_stride":CFG.MIDAS_STRIDE,
        "training_goal_directed_assistance":True,
        "assistance_uses_privileged_human_state":False,
        "assistance_end_fraction":CFG.GOAL_ASSIST_END_FRAC,
        "drive_sign":float(CFG.DRIVE_SIGN),
        "turn_sign":float(CFG.TURN_SIGN),
        "assist_behavior_cloning_coef":CFG.ASSIST_BC_COEF,
        "training_forced_exploration":True,
        "evaluation_forced_exploration":False,
        "warning":"Simulator result is not a formal real-world safety guarantee."}
    torch.save({"metadata":metadata,
                "state_dict":{k:v.detach().cpu() for k,v in proposed.state_dict().items()}},
               ROOT/"PROPOSED_LP_MDTA_RCPPO.pt")
    (ROOT/"deployment_metadata.json").write_text(json.dumps(metadata,indent=2))

    try:
        class Deploy(nn.Module):
            def __init__(self,m): super().__init__(); self.m=m
            def forward(self,vision,cmd):
                mu,_,_,_=self.m(vision,cmd)
                return torch.tanh(mu)
        wrapper=Deploy(proposed).eval()
        exv=torch.zeros(1,CFG.HISTORY*4,CFG.CAM_H,CFG.CAM_W,device=DEVICE)
        exc=torch.zeros(1,5,device=DEVICE)
        traced=torch.jit.trace(wrapper,(exv,exc))
        traced.save(str(ROOT/"PROPOSED_LP_MDTA_RCPPO_TORCHSCRIPT.pt"))
        print("TorchScript saved.")
    except Exception as exc:
        print("TorchScript export skipped:",repr(exc))




stage(16, "Final consistency checks")

assert set(TEST_CASES)==set(EVAL.case.unique())
assert len(METHODS)==8
assert float(EVAL["forced_exploration_fraction"].max())==0.0, "Evaluation must be completely unforced"
assert np.isfinite(EVAL[["return","spl","path_length_m",
                         "discomfort_fraction","stationary_fraction"]].to_numpy()).all()
assert EVAL.success.between(0,1).all()
assert EVAL.human_collision.between(0,1).all()
print("✅ Named cases present:",TEST_CASES)
print("✅ Numerical results finite")
print("✅ Binary outcome columns valid")
print(f"✅ Physics penetration audit: {'PASS' if physics_pass else 'FAIL'}")




stage(17, "Packaging all checkpoints, CSVs, plots, video and reports")

# 18) Record every output file and configuration value, zip the experiment folder, and download it automatically when running in Colab.
manifest=[]
for p in sorted(ROOT.rglob("*")):
    if p.is_file():
        manifest.append({"file":str(p.relative_to(ROOT)),"bytes":p.stat().st_size})
pd.DataFrame(manifest).to_csv(ROOT/"MANIFEST.csv",index=False)

config_dump={k:v for k,v in CFG.__dict__.items()
             if k.isupper() and isinstance(v,(str,int,float,bool))}
(ROOT/"config.json").write_text(json.dumps(config_dump,indent=2))

zip_path=shutil.make_archive(str(ROOT),"zip",root_dir=str(ROOT))
stage(18, "Complete")
print("\n"+"="*100)
print("FINISHED")
print("="*100)
print("Results:",ROOT)
print("ZIP:",zip_path)
print("Methods:",", ".join(METHODS.keys()))
print("Test cases:",", ".join(TEST_CASES))
print("8-tile comparison video:",COMPARISON_VIDEO)
print(f"MiDaS calls: {depth_model.calls:,}")
if depth_model.calls:
    print(f"Mean MiDaS time: {1000*depth_model.total_time/depth_model.calls:.2f} ms")
print("\nRead these first:")
print("  benchmark_overall_95CI.csv")
print("  benchmark_by_test_case_95CI.csv")
print("  paired_bootstrap_vs_proposed.csv")
print("  EVALUATION_REPORT.md")

try:
    from google.colab import files
    print("\nStarting browser download...")
    files.download(zip_path)
except Exception:
    print("Not in Colab. Archive remains at:",zip_path)
