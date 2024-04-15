import dm_control.suite
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

env = dm_control.suite.load(domain_name="cartpole", task_name="balance")

action_spec = env.action_spec()
time_step = env.reset()

framerate = 30

frames = []

while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    print(time_step)
    exit()
    if len(frames) < env.physics.data.time * framerate:
        pixels = env.physics.render()
        frames.append(pixels)



def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                              interval=interval, blit=True, repeat=False)
    return anim

anim = display_video(frames, framerate)
anim.save('/tmp/cartpole.gif', fps=framerate)
