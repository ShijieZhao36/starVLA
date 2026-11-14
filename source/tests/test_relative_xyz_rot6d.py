import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_sharded import _relative_chunk, _relative_xyz_rot6d


def _random_poses(rng: np.random.Generator, count: int) -> np.ndarray:
    matrices = Rotation.random(count, random_state=rng).as_matrix()
    rot6d = matrices[:, :2, :].reshape(count, 6)
    translation = rng.normal(size=(count, 3))
    return np.concatenate((translation, rot6d), axis=-1).astype(np.float32)


class RelativeXyzRot6dTest(unittest.TestCase):
    def test_batch_matches_pose_subtraction(self):
        rng = np.random.default_rng(0)
        reference = _random_poses(rng, 5)
        windows = _random_poses(rng, 5 * 8).reshape(5, 8, 9)
        config = ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.EEF,
            format=ActionFormat.XYZ_ROT6D,
            state_key="left_eef",
        )
        expected = np.stack(
            [
                _relative_chunk(windows[index], reference[index], config)
                for index in range(len(reference))
            ],
            axis=0,
        )
        actual = _relative_xyz_rot6d(windows, reference)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
