import collections
import os
import warnings

import cv2
import gymnasium as gym
import numpy as np

with warnings.catch_warnings():
    # Filter out DeprecationWarnings raised from pkg_resources
    warnings.filterwarnings(
        "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
    )
    import pygame

import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from .pymunk_override import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushT2Env(gym.Env):
    """
    ## Description

    PushT2 environment.

    The goal of the agent is to push the block to two goal zones, one after the other. The agent is a circle and the block is a tee shape.
    The two goals are fixed at upper left and bottom right positions.

    ## Action Space

    The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
    represent the target position of the agent.

    ## Observation Space

    If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
    environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
    and block positions and [0, 2*pi] for the block angle.

    If `obs_type` is set to `environment_state_agent_pos` the observation space is a dictionary with:
    - `environment_state`: 16-dimensional vector representing the keypoint locations of the T (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, 512]. See `get_keypoints` for a diagram showing the location of the
        keypoint indices.
    - `agent_pos`: A 2-dimensional vector representing the position of the robot end-effector.

    If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

    ## Rewards

    The reward is 0.5 for each goal zone covered (first time only). Maximum reward is 1.0 when both goals are reached.

    ## Success Criteria

    The environment is considered solved if both goal zones have been covered by at least 90% each.

    ## Starting State

    The agent starts at a random position and the block starts at a random position and angle.

    ## Episode Termination

    The episode terminates when both goal zones have been covered by at least 90% each.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
    ```

    * `obs_type`: (str) The observation type. Can be either `state`, `keypoints`, `pixels` or `pixels_agent_pos`.
      Default is `state`.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `observation_width`: (int) The width of the observed image. Default is `96`.

    * `observation_height`: (int) The height of the observed image. Default is `96`.

    * `visualization_width`: (int) The width of the visualized image. Default is `680`.

    * `visualization_height`: (int) The height of the visualized image. Default is `680`.

    * `success_threshold`: (float) The coverage threshold (0.0 to 1.0) required for a goal to be considered reached.
      Default is `0.90` (90% coverage).

    * `visualize_goal_progress`: (bool) Whether to visualize which goals have been reached by changing their color.
      If `True`, reached goals turn blue. If `False`, all goals remain green. Default is `False` to prevent the
      policy from observing goal progress through pixel observations.

    ## Reset Arguments

    Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

    > [!WARNING]
    > For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
    > the one passed in the argument.

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0")
    >>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
    >>> state
    array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
          dtype=float32)
    ```

    ## Version History

    * v0: Original version

    ## References

    * TODO:
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        block_cog=None,
        damping=None,
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
        success_threshold=0.90,
        visualize_goal_progress=False,
    ):
        super().__init__()
        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Initialize spaces
        self._initialize_observation_space()
        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = (
            success_threshold  # 90% coverage for each goal by default
        )
        self.visualize_goal_progress = (
            visualize_goal_progress  # Whether to show visual feedback of reached goals
        )

        # Track which goals have been reached
        self.goal_1_reached = False
        self.goal_2_reached = False

    def _initialize_observation_space(self):
        if self.obs_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0]),
                high=np.array([512, 512, 512, 512, 2 * np.pi]),
                dtype=np.float64,
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "environment_state": spaces.Box(
                        low=np.zeros(16),
                        high=np.full((16,), 512),
                        dtype=np.float64,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                },
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, environment_state_agent_pos, "
                "pixels_agent_pos]"
            )

    def _get_coverage(self, goal_pose):
        goal_body = self.get_goal_pose_body(goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return intersection_area / goal_area

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)

        # Compute coverage for both goals
        coverage_1 = self._get_coverage(self.goal_pose_1)
        coverage_2 = self._get_coverage(self.goal_pose_2)

        # Update goal reached flags and calculate reward
        reward = 0.0
        if coverage_1 > self.success_threshold and not self.goal_1_reached:
            self.goal_1_reached = True
            reward += 0.5
        if coverage_2 > self.success_threshold and not self.goal_2_reached:
            self.goal_2_reached = True
            reward += 0.5

        # Success only when both goals are reached
        is_success = self.goal_1_reached and self.goal_2_reached
        terminated = is_success

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["coverage_goal_1"] = coverage_1
        info["coverage_goal_2"] = coverage_2
        info["goal_1_reached"] = self.goal_1_reached
        info["goal_2_reached"] = self.goal_2_reached

        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()

        # Reset goal tracking
        self.goal_1_reached = False
        self.goal_2_reached = False

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            # we use the env's RNG instead of global np.random
            state = np.array(
                [
                    self.np_random.integers(50, 450),
                    self.np_random.integers(50, 450),
                    self.np_random.integers(100, 400),
                    self.np_random.integers(100, 400),
                    self.np_random.uniform(-np.pi, np.pi),
                ]
            )
        self._set_state(state)

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = False

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)

        # Draw both goal poses
        for goal_pose, reached in [
            (self.goal_pose_1, self.goal_1_reached),
            (self.goal_pose_2, self.goal_2_reached),
        ]:
            goal_body = self.get_goal_pose_body(goal_pose)
            # Only show different colors for reached goals if visualize_goal_progress is True
            if self.visualize_goal_progress:
                color = (
                    pygame.Color("LightBlue") if reached else pygame.Color("LightGreen")
                )
            else:
                color = pygame.Color("LightGreen")  # Always show same color
            for shape in self.block.shapes:
                goal_points = [
                    goal_body.local_to_world(v) for v in shape.get_vertices()
                ]
                goal_points = [
                    pymunk.pygame_util.to_pygame(point, draw_options.surface)
                    for point in goal_points
                ]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(screen, color, goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self._last_action is not None:
            action = np.array(self._last_action)
            coord = (action / 512 * [height, width]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()  # draw the environment on a screen

        if self.render_mode == "rgb_array":
            return self._get_img(
                screen, width=width, height=height, render_action=visualize
            )
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(
                screen, screen.get_rect()
            )  # copy our drawings from `screen` to the visible window
            pygame.event.pump()
            self.clock.tick(
                self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz))
            )
            pygame.display.update()
        else:
            raise ValueError(self.render_mode)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            # Create a temporary screen surface for coordinate conversion
            screen = pygame.Surface((512, 512))
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate(
                [agent_position, block_position, [block_angle]], dtype=np.float64
            )

        if self.obs_type == "environment_state_agent_pos":
            return {
                "environment_state": self.get_keypoints(self._block_shapes).flatten(),
                "agent_pos": np.array(self.agent.position),
            }

        pixels = self._render()
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.array(self.agent.position),
            }

    @staticmethod
    def get_goal_pose_body(pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose_1": self.goal_pose_1,
            "goal_pose_2": self.goal_pose_2,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False

        # Add walls
        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zones
        self.agent = self.add_circle(self.space, (256, 400), 15)
        self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)

        # Two fixed goal poses: upper left and bottom right
        self.goal_pose_1 = np.array(
            [200, 150, np.pi / 4]
        )  # upper left: x, y, theta (in radians)
        self.goal_pose_2 = np.array(
            [350, 300, 0]
        )  # bottom right: x, y, theta (in radians)

        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _set_state(self, state):
        self.agent.position = list(state[:2])
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoretically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        self.block.position = list(state[2:4])
        self.block.angle = state[4]

        # Run physics to take effect
        self.space.step(self.dt)

    @staticmethod
    def add_segment(space, a, b, radius):
        # TODO(rcadene): rename add_segment to make_segment, since it is not added to the space
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    @staticmethod
    def add_circle(space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        space.add(body, shape)
        return body

    @staticmethod
    def add_tee(space, position, angle, scale=30, color="LightSlateGray", mask=None):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.angle = angle
        body.position = position
        body.friction = 1
        space.add(body, shape1, shape2)
        return body, [shape1, shape2]

    @staticmethod
    def get_keypoints(block_shapes):
        """Get a (8, 2) numpy array with the T keypoints.

        The T is composed of two rectangles each with 4 keypoints.

        0───────────1
        │           │
        3───4───5───2
            │   │
            │   │
            │   │
            │   │
            7───6
        """
        keypoints = []
        for shape in block_shapes:
            for v in shape.get_vertices():
                v = v.rotated(shape.body.angle)
                v = v + shape.body.position
                keypoints.append(np.array(v))
        return np.row_stack(keypoints)
