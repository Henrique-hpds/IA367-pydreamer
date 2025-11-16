import threading
import numpy as np

from queue import Queue

class AlphaScheduler:

    def __init__(self, alpha_init: float = 1.0, alpha_final: float = 0.1, tau: float = 100000.0):
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.tau = tau
        self.step_count = 0
        
    def step(self) -> float:
        alpha = self.get_alpha()
        self.step_count += 1
        return alpha
    
    def get_alpha(self) -> float:
        if self.tau <= 0:
            return self.alpha_final
        alpha = self.alpha_init * np.exp(-self.step_count / self.tau)
        return max(self.alpha_final, alpha)
    
    def reset(self):
        self.step_count = 0
        
    def set_step(self, step: int):
        self.step_count = step


class EnvironmentCollector:
    def __init__(self, collector_id: int, env_fn, policy_fn, queue, steps_per_send: int = 50, sequence_length: int = 50):
        self.collector_id = collector_id
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.queue = queue
        self.steps_per_send = steps_per_send
        self.sequence_length = sequence_length
        self.running = False
        
    def run(self):
        self.running = True
        env = self.env_fn()
        
        # Initialize buffers
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        terminal_buffer = []
        reset_buffer = []
        
        obs = env.reset()
        state = None  # Policy state
        steps = 0
        
        while self.running:
            # Get action from policy
            action, state = self.policy_fn(obs, state)
            
            # Store transition
            obs_buffer.append(obs)
            action_buffer.append(action)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            reward_buffer.append(reward)
            terminal_buffer.append(done)
            reset_buffer.append(False)  # Will be set to True at episode start
            
            obs = next_obs
            steps += 1
            
            # Reset on episode end
            if done:
                obs = env.reset()
                state = None
                # Mark next step as reset
                if len(reset_buffer) > 0:
                    reset_buffer[-1] = True
            
            # Send sequences when buffer is full
            if len(obs_buffer) >= self.sequence_length:
                sequence = self._create_sequence(
                    obs_buffer[:self.sequence_length],
                    action_buffer[:self.sequence_length],
                    reward_buffer[:self.sequence_length],
                    terminal_buffer[:self.sequence_length],
                    reset_buffer[:self.sequence_length]
                )
                self.queue.put(sequence)
                
                # Keep overlap for continuity
                overlap = self.sequence_length // 4
                obs_buffer = obs_buffer[self.sequence_length - overlap:]
                action_buffer = action_buffer[self.sequence_length - overlap:]
                reward_buffer = reward_buffer[self.sequence_length - overlap:]
                terminal_buffer = terminal_buffer[self.sequence_length - overlap:]
                reset_buffer = reset_buffer[self.sequence_length - overlap:]
        
        env.close()
    
    def _create_sequence(self, obs_list, action_list, reward_list, terminal_list, reset_list):
        # Convert observations
        if isinstance(obs_list[0], dict):
            obs_dict = {}
            for key in obs_list[0].keys():
                obs_dict[key] = np.stack([o[key] for o in obs_list], axis=0)
            sequence = obs_dict
        else:
            sequence = {'observation': np.stack(obs_list, axis=0)}
        
        # Add other data
        sequence['action'] = np.stack(action_list, axis=0)
        sequence['reward'] = np.array(reward_list, dtype=np.float32)
        sequence['terminal'] = np.array(terminal_list, dtype=bool)
        sequence['reset'] = np.array(reset_list, dtype=bool)
        
        return sequence
    
    def stop(self):
        self.running = False


class DistributedCollectorManager:    
    def __init__(self, num_collectors: int, env_fn, policy_fn, steps_per_send: int = 50, sequence_length: int = 50, queue_maxsize: int = 100):

        self.num_collectors = num_collectors
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.steps_per_send = steps_per_send
        self.sequence_length = sequence_length
        
        self.queue = Queue(maxsize=queue_maxsize)
        
        self.collectors = []
        self.threads = []
        
    def start(self):
        for i in range(self.num_collectors):
            collector = EnvironmentCollector(
                collector_id=i,
                env_fn=self.env_fn,
                policy_fn=self.policy_fn,
                queue=self.queue,
                steps_per_send=self.steps_per_send,
                sequence_length=self.sequence_length
            )
            self.collectors.append(collector)
            
            thread = threading.Thread(target=collector.run, daemon=True)
            thread.start()
            self.threads.append(thread)
    
    def stop(self):
        for collector in self.collectors:
            collector.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
    
    def get_queue(self):
        return self.queue
