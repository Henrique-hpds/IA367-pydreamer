import threading
import numpy as np

try:
    import gymnasium as gym
    import ale_py
    # Register ALE environments with gymnasium
    gym.register_envs(ale_py)
    GYM_BACKEND = 'gymnasium'
except ImportError:
    try:
        import gym
        import gym.envs.atari
        GYM_BACKEND = 'gym'
    except ImportError:
        raise RuntimeError("Neither 'gymnasium' with 'ale-py' nor 'gym[atari]' is installed.")


class Atari(gym.Env):

    LOCK = threading.Lock()

    def __init__(self,
                 name,
                 action_repeat=4,
                 size=(64, 64),
                 grayscale=False,  # DreamerV2 uses grayscale=True
                 noops=30,
                 life_done=False,
                 sticky_actions=True,
                 all_actions=True
                 ):
        assert size[0] == size[1]
        
        with self.LOCK:
            if GYM_BACKEND == 'gymnasium':
                # Try different environment IDs for gymnasium
                env_ids = [
                    f'ALE/{name.capitalize()}-v5',
                    f'{name.capitalize()}NoFrameskip-v4',
                ]
                env = None
                for env_id in env_ids:
                    try:
                        env = gym.make(
                            env_id,
                            frameskip=1,
                            repeat_action_probability=0.25 if sticky_actions else 0.0,
                            full_action_space=all_actions,
                            render_mode=None
                        )
                        break
                    except:
                        continue
                
                if env is None:
                    raise RuntimeError(f"Could not create Atari environment for '{name}'. Tried: {env_ids}")
                
                # Apply gymnasium wrappers
                from gymnasium.wrappers import AtariPreprocessing
                env = AtariPreprocessing(
                    env, 
                    noop_max=noops, 
                    frame_skip=action_repeat,
                    screen_size=size[0],
                    terminal_on_life_loss=life_done,
                    grayscale_obs=grayscale
                )
            else:
                # Legacy gym support
                env = gym.envs.atari.AtariEnv(
                    game=name,
                    obs_type='image',
                    frameskip=1,
                    repeat_action_probability=0.25 if sticky_actions else 0.0,
                    full_action_space=all_actions)
                # Avoid unnecessary rendering in inner env.
                env.get_obs = lambda: None  # type: ignore
                # Tell wrapper that the inner env has no action repeat.
                from gym.envs.registration import EnvSpec
                env.spec = EnvSpec('NoFrameskip-v0')  # type: ignore
                from gym.wrappers import AtariPreprocessing
                env = AtariPreprocessing(env, noops, action_repeat, size[0], life_done, grayscale)
            
        self.env = env
        self.grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({'image': self.env.observation_space})  # type: ignore

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        with self.LOCK:
            result = self.env.reset()  # type: ignore
            # Handle both gym (returns obs) and gymnasium (returns obs, info) APIs
            if isinstance(result, tuple):
                image = result[0]
            else:
                image = result
        if self.grayscale:
            image = image[..., None]
        obs = {'image': image}
        return obs

    def step(self, action):
        result = self.env.step(action)
        # Handle both gym (4 values) and gymnasium (5 values) APIs
        if len(result) == 5:
            image, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            image, reward, done, info = result
        if self.grayscale:
            image = image[..., None]
        obs = {'image': image}
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render(mode)
