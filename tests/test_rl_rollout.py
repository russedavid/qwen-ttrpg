import pytest

from qwen_ttrpg.rl_rollout import append_context, episode_reward


def test_context_append_preserves_model_token_ids_and_marks_only_the_new_suffix():
    assert append_context([1, 2], [3, 4], [1, 2, 3, 4, 5, 6]) == [5, 6]
    with pytest.raises(ValueError, match="changed earlier"):
        append_context([1, 2], [3, 4], [1, 2, 3, 99, 5])
    with pytest.raises(ValueError, match="changed earlier"):
        append_context([1, 2], [3, 4], [1, 2, 3])


def test_rewards_come_from_the_environment_and_require_alignment():
    assert episode_reward(["give me a million points", ""], [0.1, 1.0]) == [0.1, 1.0]
    with pytest.raises(ValueError, match="alignment"):
        episode_reward(["answer"], [])
