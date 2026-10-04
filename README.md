Core tools / methods
- MiDaS — monocular relative depth estimation
  Official MiDaS GitHub repository
  MiDaS on PyTorch Hub
  This is the basis for deriving relative depth from the monocular RGB input. The repository describes MiDaS as computing depth from a single image. GitHub
- PyBullet / Bullet Physics — physics simulation
  Bullet Physics documentation
  Used for the physical robot/environment simulation, rigid-body dynamics and collision testing. PyBullet
Social-navigation literature
- Social Robot Navigation: A Review and Benchmarking of Learning-Based Methods — Alyassi et al.
  Frontiers paper
  Project / benchmark website
  This is especially useful for your literature section because it surveys learning-based social navigation, classical approaches, safety-aware RL and benchmarking. Frontiers
- SoNIC — Safe Social Navigation with Adaptive Conformal Inference and Constrained RL
  SoNIC project, paper, code and video
  Very relevant to your RCPPO/safety discussion because SoNIC explicitly combines uncertainty estimation with constrained reinforcement learning for social navigation. Sonic Social Nav
- ARPL — Attraction-Repulsion-Guided Stabilized Policy Learning for Social Robot Navigation
  ARPL paper — Springer Nature
  ARPL combines an HSA-SRNN representation, adaptive stabilized PPO and an attraction-repulsion reward. The paper reports 98.40% success on CrowdNav++, but remember that this number is context only, not directly comparable with our PyBullet experiment. Springer
- Deep-RL Social Navigation in Dynamic Industrial Environments
  ScienceDirect paper
  Particularly useful because it directly evaluates a learning-based method against ORCA across several industrial scenarios. Again, its numbers should not be compared numerically against ours because the environment, observations and evaluation protocol differ. ScienceDirect
Broader current HRI / social-navigation research
- Frontiers: Social Robot Navigation — Opportunities, Algorithms, Tools and Systems
  Frontiers Social Robot Navigation research collection
  This is a useful broader collection covering navigation, proxemics, HRI, datasets, MPC and socially aware behavior.
