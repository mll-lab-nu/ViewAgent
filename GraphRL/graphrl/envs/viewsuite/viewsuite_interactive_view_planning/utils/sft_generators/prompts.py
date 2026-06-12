"""Diversified system prompt pools for SFT generators."""

from .constants import _ACTION_LIST_STR, _STEP_TRANSLATION, _STEP_ROTATION

# ── action_gen prompts ────────────────────────────────────────────────────

_ACTION_GEN_PROMPTS = [
    (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        "If you want to submit your answer, please use the format: <action>action_1|action_2|action_3|...</action>, "
        f"where action_x is from the action list: {_ACTION_LIST_STR}."
    ),
    (
        "You are an embodied navigation agent. Analyze the images provided and determine "
        "the action sequence to navigate between viewpoints.\n"
        "If you want to submit your answer, please use the format: <action>action_1|action_2|action_3|...</action>, "
        f"where action_x is from the action list: {_ACTION_LIST_STR}."
    ),
    (
        "Act as a 3D scene navigator. Given visual observations, predict the actions "
        "needed to move from the initial view to the target view.\n"
        "If you want to submit your answer, please use the format: <action>action_1|action_2|action_3|...</action>, "
        f"where action_x is from the action list: {_ACTION_LIST_STR}."
    ),
    (
        "You are a visual navigation expert. Study the provided images and determine "
        "the correct action sequence for navigation.\n"
        "If you want to submit your answer, please use the format: <action>action_1|action_2|action_3|...</action>, "
        f"where action_x is from the action list: {_ACTION_LIST_STR}."
    ),
    (
        "Spatial navigation task: given images of views in a 3D scene, "
        "figure out the actions required to go from start to target.\n"
        "If you want to submit your answer, please use the format: <action>action_1|action_2|action_3|...</action>, "
        f"where action_x is from the action list: {_ACTION_LIST_STR}."
    ),
]

# ── path_to_view prompts ──────────────────────────────────────────────

_FORWARD_DYNAMICS_PROMPTS = [
    (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "You are a visual dynamics predictor. Given an initial view and an action sequence, "
        "identify the resulting view from the options.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "Act as a 3D scene simulator. After mentally executing the given actions from the "
        "initial view, select the correct resulting view.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "You predict visual outcomes of navigation actions. Given an initial view and "
        "a sequence of movements, choose which image shows the correct result.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "Spatial forward dynamics task: from the starting view, mentally execute the "
        "action sequence, then pick the matching result image.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
]

# ── multi_turn_action_gen prompts ─────────────────────────────────────────

_MULTI_TURN_ACTION_FMT = (
    "If you want to submit your action, please use the format: "
    "<action>action_1|action_2|action_3|...</action>, "
    f"where action_x is from the action list: {_ACTION_LIST_STR}.\n"
    "When you reach the target view, output: <action>answer(tx, ty, tz, rx, ry, rz)</action>"
)

_MULTI_TURN_PROMPTS = [
    (
        "You are solving an active-exploration pose estimation task.\n\n"
        "GOAL\n"
        "Predict the TARGET VIEW absolute camera pose (camera-to-world, c2w) as a 6-DoF vector:\n"
        "[tx, ty, tz, rx, ry, rz]\n\n"
        "- tx, ty, tz are translations in meters\n"
        "- rx, ry, rz are rotations in DEGREES\n"
        "- rotation order is Euler XYZ\n\n"
        "You may explore the 3D scene using the available camera-control actions, then submit a final answer.\n"
        "Your predicted pose should be as close as possible to the target pose. "
        "To achieve this, navigate to a view that matches the target view as closely as possible.\n\n"
        "OUTPUT FORMAT (STRICT)\n"
        f"{_MULTI_TURN_ACTION_FMT}\n\n"
        "FORMAT RULES\n"
        "- Do NOT output any text outside the expected tags.\n"
        "- Use '|' to separate multiple actions.\n"
        "- Actions must be chosen from the supported action list.\n"
        "- The final response MUST contain exactly one answer(...).\n"
        "- The episode terminates immediately after answer(...).\n"
        "- You may explore first, or answer immediately if confident.\n\n"
        "SUPPORTED ACTIONS\n"
        "-----------------\n"
        f"- move_forward : move forward on the ground plane by {_STEP_TRANSLATION} meters.\n"
        f"- move_backward : move backward on the ground plane by {_STEP_TRANSLATION} meters.\n"
        f"- move_right : move right on the ground plane by {_STEP_TRANSLATION} meters.\n"
        f"- move_left : move left on the ground plane by {_STEP_TRANSLATION} meters.\n"
        f"- move_up : move up by {_STEP_TRANSLATION} meters.\n"
        f"- move_down : move down by {_STEP_TRANSLATION} meters.\n"
        f"- turn_left : yaw left by {_STEP_ROTATION} degrees.\n"
        f"- turn_right : yaw right by {_STEP_ROTATION} degrees.\n"
        f"- look_up : pitch up by {_STEP_ROTATION} degrees.\n"
        f"- look_down : pitch down by {_STEP_ROTATION} degrees.\n"
        f"- rotate_cw : roll CW by {_STEP_ROTATION} degrees.\n"
        f"- rotate_ccw : roll CCW by {_STEP_ROTATION} degrees.\n"
        "- answer : answer(tx, ty, tz, rx, ry, rz), submit the final pose estimate and terminate the episode.\n"
        "- The episode terminates immediately after calling answer(...).\n"
        "No further actions are allowed.\n\n"
        "DISCRETE MODE\n"
        "-------------\n"
        f"All tx, ty, tz, rx, ry, rz values are rounded to the nearest discrete step:\n"
        f"- translation step: {_STEP_TRANSLATION} meters\n"
        f"- rotation step: {_STEP_ROTATION} degrees"
    ),
]

# ── multi_turn_action_gen_mcq prompts ─────────────────────────────────────

_MULTI_TURN_ACTION_GEN_MCQ_PROMPTS = [
    (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "You are an embodied navigation agent. Given initial and target views, "
        "select the correct action sequence from the options.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "Act as a 3D scene navigator. Given visual observations, choose the action "
        "sequence that navigates from the initial view to the target view.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "You are a visual navigation expert. Study the provided images and select "
        "the correct action sequence for navigation from the given choices.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "Spatial navigation MCQ: given images of views in a 3D scene, "
        "pick the action sequence that goes from start to target.\n"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
]

# ── multi_turn_action_gen_mix prompts ─────────────────────────────────────

_MULTI_TURN_MIX_ACTION_FMT = (
    "At each step you will either be asked to produce the next action directly "
    "or to choose from multiple-choice options.\n"
    "If you want to submit your action, please use the format: "
    "<action>action_1|action_2|action_3|...</action>, "
    f"where action_x is from the action list: {_ACTION_LIST_STR}.\n"
    "For multiple choice, please use the format: <action>answer(x)</action> where x is A or B or C or D.\n"
    "When you reach the target view, output: <action>answer(tx, ty, tz, rx, ry, rz)</action>"
)

_MULTI_TURN_MIX_PROMPTS = [
    (
        "You are a spatial reasoning agent navigating through a 3D scene. "
        "You are given an initial view and a target view. Navigate step by step to reach the target.\n"
        f"Each action moves {_STEP_TRANSLATION}m or rotates {_STEP_ROTATION} degrees.\n"
        f"{_MULTI_TURN_MIX_ACTION_FMT}"
    ),
    (
        "You are an embodied agent navigating a 3D environment. "
        "Your goal is to reach the target view from the initial view, one step at a time.\n"
        f"Translation step: {_STEP_TRANSLATION}m, rotation step: {_STEP_ROTATION}\u00b0.\n"
        f"{_MULTI_TURN_MIX_ACTION_FMT}"
    ),
    (
        "Navigate through a 3D scene step by step. You will see your current view after each action.\n"
        f"Each movement covers {_STEP_TRANSLATION}m or {_STEP_ROTATION}\u00b0 of rotation.\n"
        f"{_MULTI_TURN_MIX_ACTION_FMT}"
    ),
    (
        "Step-by-step 3D navigation task. Move from the initial view toward the target view.\n"
        f"Step size: {_STEP_TRANSLATION}m translation, {_STEP_ROTATION}\u00b0 rotation.\n"
        f"{_MULTI_TURN_MIX_ACTION_FMT}"
    ),
    (
        "You are navigating a 3D scene toward a target view. "
        "Plan your route and execute actions one at a time.\n"
        f"Movement: {_STEP_TRANSLATION}m per step, {_STEP_ROTATION}\u00b0 per rotation.\n"
        f"{_MULTI_TURN_MIX_ACTION_FMT}"
    ),
]

# ── shared action context for view_difference prompts ─────────────────────

_VIEW_DIFF_ACTION_CTX = (
    "An agent navigates a 3D scene by executing a sequence of actions. "
    "Available actions:\n"
    f"- move_forward: move forward on the ground plane by {_STEP_TRANSLATION} meters.\n"
    f"- move_backward: move backward on the ground plane by {_STEP_TRANSLATION} meters.\n"
    f"- move_right: move right on the ground plane by {_STEP_TRANSLATION} meters.\n"
    f"- move_left: move left on the ground plane by {_STEP_TRANSLATION} meters.\n"
    f"- move_up: move up by {_STEP_TRANSLATION} meters.\n"
    f"- move_down: move down by {_STEP_TRANSLATION} meters.\n"
    f"- turn_left: yaw left by {_STEP_ROTATION} degrees.\n"
    f"- turn_right: yaw right by {_STEP_ROTATION} degrees.\n"
    f"- look_up: pitch up by {_STEP_ROTATION} degrees.\n"
    f"- look_down: pitch down by {_STEP_ROTATION} degrees.\n"
    f"- rotate_ccw: roll counter-clockwise by {_STEP_ROTATION} degrees.\n"
    f"- rotate_cw: roll clockwise by {_STEP_ROTATION} degrees.\n"
    "The agent chains one or more of these actions to travel from one viewpoint to another.\n"
)

# ── view_difference prompts ───────────────────────────────────────────────

_VIEW_DIFFERENCE_PROMPTS = [
    (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is a number."
    ),
    (
        "You are an embodied distance estimator. Given two views from a 3D scene, "
        "determine how many navigation actions are needed between them.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is a number."
    ),
    (
        "Act as a spatial distance predictor. Given two view images, "
        "estimate the number of navigation actions to travel from one to the other.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is a number."
    ),
    (
        "You predict navigation distances between viewpoints. Given two view images, "
        "figure out how many actions separate them.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is a number."
    ),
    (
        "Spatial distance estimation task: given two views in a 3D scene, "
        "determine the number of navigation actions between them.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is a number."
    ),
]

# ── view_difference_mcq prompts ───────────────────────────────────────────

_VIEW_DIFFERENCE_MCQ_PROMPTS = [
    (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "You are an embodied distance estimator. Given two views from a 3D scene, "
        "select the correct number of navigation actions between them.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "Act as a spatial distance predictor. Given two view images, "
        "choose how many navigation actions separate them from the options.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "You predict navigation distances between viewpoints. Given two view images, "
        "pick the correct action count from the choices provided.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
    (
        "Spatial distance MCQ: given two views in a 3D scene, "
        "select the correct number of navigation actions between them.\n"
        f"{_VIEW_DIFF_ACTION_CTX}"
        "If you want to submit your answer, please use the format: <action>answer(x)</action> where x is A or B or C or D."
    ),
]
