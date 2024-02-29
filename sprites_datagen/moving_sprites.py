import numpy as np
import cv2
from torch.utils.data import Dataset

from general_utils import AttrDict
from sprites_datagen.utils.template_blender import TemplateBlender
from sprites_datagen.utils.trajectory import ConstantSpeedTrajectory

class MovingSpriteDataset(Dataset):
    """Dataset of multiple sprites bouncing in frame, contains different reward annotations."""
    # 
    def __init__(self, spec):
        self._spec = spec
        # COMMON for two objects as least (Agent+Target)
        self._generator = DistractorTemplateMovingSpritesGenerator(self._spec)
        # SPECIAL for one object
        if self._spec.shapes_per_traj <= 1:
            self._generator = TemplateMovingSpritesGenerator(self._spec)

    # 
    def __len__(self):
        return self._spec.dataset_size

    # 
    def __getitem__(self, item):
        traj = self._generator.gen_trajectory()

        data_dict = AttrDict()
        
        # IMAGEs
        if self._spec.input_channels == 3 :
            data_dict.images = traj.images[:, None].repeat(3, axis=1).astype(np.float32) / (255./2) - 1.0 # [T, 3, resolution, resolution] -> [-1,1]
        elif self._spec.input_channels == 1 :
            data_dict.images = traj.images[:, None].astype(np.float32) / 255. # [T, 1, resolution, resolution] -> [0,1]
        else : 
            data_dict.images = traj.images[:, None].astype(np.float32) / 255. # [T, 1, resolution, resolution] -> [0,1]

        data_dict.states = traj.states          # [T, shapes_per_traj, 2(x,y)]
        data_dict.shape_idxs = traj.shape_idxs  # [ AGENT(1) | TARGET(0) | Distractors(...)]
        data_dict.rewards = traj.rewards        # [max_seq_len]

        return data_dict

class MovingSpritesGenerator:
    """Base moving sprites data generator class."""
    SHAPES = ['rectangle', 'circle', 'tri_right', 'tri_bottom', 'tri_left', 'tri_top']

    def __init__(self, spec):
        self._spec = spec
        bounds = [[self._spec.obj_size/2, 1 - self._spec.obj_size/2]] * 2
        self._traj_gen = ConstantSpeedTrajectory(n_dim=2, pos_bounds=bounds, max_speed=self._spec.max_speed)

    def gen_trajectory(self):
        """Samples trajectory with bouncing sprites."""
        output = AttrDict()

        # sample coordinate trajectories [T, n_shapes, state_dim]
        output.states = self._traj_gen.create(self._spec.max_seq_len, self._spec.shapes_per_traj) # Generate Trajectories

        # sample shapes for trajectory
        output.shape_idxs = self._sample_shapes()              # [ AGENT(1) | TARGET(0) | Distractors(...)]
        shapes = np.asarray(self.SHAPES)[output.shape_idxs]    # the shapes'names accroding to the indexes

        # render images for trajectories + shapes
        output.images = self._render(output.states, shapes)    # ['Circle' | 'rectangle' | Distractors(...)]

        # compute rewards for trajectories
        output.rewards = self._reward(output.states, shapes)

        return output

    def _sample_shapes(self):
        """Randomly samples shapes from the set of available shapes. Can be overwritten in inheriting classes."""
        return np.random.choice(np.arange(len(self.SHAPES)), size=self._spec.shapes_per_traj)

    def _reward(self, trajectories, shapes):
        """Computes the reward for a given trajectory."""
        return {r_class().name: r_class()(trajectories, shapes) for r_class in self._spec.rewards}

    def _render(self, trajectories, shapes):
        """Renders a given state trajectory."""
        raise NotImplementedError


class TemplateMovingSpritesGenerator(MovingSpritesGenerator):
    """Moving sprites filled in with template sprites in 2D."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sprite_res = int(self._spec.obj_size * self._spec.resolution)
        self._shape_sprites = self._get_shape_sprites()  # generate geometric shape templates
        self._template_blender = TemplateBlender((self._spec.resolution, self._spec.resolution))

    def _render(self, trajectories, shapes):
        sprites = [self._shape_sprites[shape] for shape in shapes]
        return self._template_blender.create((trajectories * (self._spec.resolution - 1)).astype(int), sprites)

    def _get_shape_sprites(self):
        shapes = AttrDict()
        canvas = np.zeros((self._sprite_res, self._sprite_res), np.uint8)
        shapes.rectangle = cv2.rectangle(canvas.copy(), (1, 1), (self._sprite_res - 2, self._sprite_res - 2), 255, -1)
        shapes.circle = cv2.circle(canvas.copy(), (int(self._sprite_res / 2), int(self._sprite_res / 2)),
                                   int(self._sprite_res / 3), 255, -1)
        shapes.tri_right = cv2.fillConvexPoly(canvas.copy(),
                                              np.array([[[1, 1], [1, self._sprite_res - 2],
                                                         [self._sprite_res - 2, int(self._sprite_res / 2)]]]), 255)
        shapes.tri_bottom = cv2.fillConvexPoly(canvas.copy(),
                                               np.array([[[1, 1], [self._sprite_res - 2, 1],
                                                          [int(self._sprite_res / 2), self._sprite_res - 2]]]), 255)
        shapes.tri_left = cv2.fillConvexPoly(canvas.copy(),
                                             np.array([[[self._sprite_res - 2, 1], [self._sprite_res - 2, self._sprite_res - 2],
                                                        [1, int(self._sprite_res / 2)]]]), 255)
        shapes.tri_top = cv2.fillConvexPoly(canvas.copy(),
                                            np.array([[[1, self._sprite_res - 2], [self._sprite_res - 2, self._sprite_res - 2],
                                                       [int(self._sprite_res / 2), 1]]]), 255)
        return shapes


class DistractorTemplateMovingSpritesGenerator(TemplateMovingSpritesGenerator):
    """Differentiates between agent, target and distractor shapes."""
    AGENT = 'circle'
    TARGET = 'rectangle'

    def _sample_shapes(self):
        """Retrieves shapes for agent and target, samples randomly from other shapes for distractors."""
        assert self._spec.shapes_per_traj >= 2

        # agent + target
        shape_idxs = np.asarray([self.SHAPES.index(self.AGENT), self.SHAPES.index(self.TARGET)])
        # 
        distractor_idxs = np.setdiff1d(np.arange(len(self.SHAPES)), shape_idxs)

        # distractors
        if self._spec.shapes_per_traj > 2:
            shape_idxs = np.concatenate((shape_idxs,
                                         np.random.choice(distractor_idxs, size=self._spec.shapes_per_traj - 2)))
        return shape_idxs


if __name__ == '__main__':
    import cv2
    from general_utils import make_image_seq_strip
    from sprites_datagen.rewards import ZeroReward

    spec = AttrDict(
        resolution=128,             # the quality of an image
        max_seq_len=30,             # the length of a sequence
        max_speed=0.05,             # total image range [0, 1]
        obj_size=0.2,               # size of objects, full images is 1.0
        shapes_per_traj=4,          # number of shapes per trajectory
        rewards=[ZeroReward],
    )
    
    gen = DistractorTemplateMovingSpritesGenerator(spec)
    traj = gen.gen_trajectory()
    img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
    
    cv2.imwrite("test.png", img[0].transpose(1, 2, 0))


