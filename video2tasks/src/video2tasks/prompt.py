def prompt_switch_detection(n_images: int, mode: str = "coarse") -> str:
    """Build the VLM prompt for the selected segmentation mode."""
    mode = (mode or "coarse").lower()
    if mode in {"fine", "action_phase"}:
        return prompt_action_phase_detection(n_images)
    return prompt_object_switch_detection(n_images)


def _strict_json_suffix() -> str:
    return (
        "### Output Format: Strict JSON\n"
        "Return only a valid JSON object. Do not include markdown fences.\n"
        "The JSON object must contain:\n"
        "- \"thought\": concise reasoning about the visible manipulation phases.\n"
        "- \"transitions\": a sorted list of image indices where a new segment starts.\n"
        "- \"instructions\": one short verb phrase for each segment. The number of instructions must equal len(transitions) + 1.\n"
        "Use image indices, not timestamps. Valid indices are within the sampled image range.\n"
    )


def prompt_object_switch_detection(n_images: int) -> str:
    """Coarse mode: split only when the active manipulated object changes."""
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"Mapping: image indices range from 0 to {n_images - 1}.\n\n"
        "### Goal\n"
        "Detect coarse atomic task boundaries. A switch occurs strictly when the robot completes interaction with one object "
        "and starts interacting with a different object.\n\n"
        "### Core Logic: Distinct Object Rule\n"
        "1. True switch: the robot releases Object A and moves to grasp/manipulate Object B. Mark the first image where the new object interaction begins.\n"
        "2. False switch: manipulating different parts of the same object is not a switch. Folding sleeves and then the body of the same shirt remains one segment.\n"
        "3. Regrasping, camera motion, temporary occlusion, or small pose adjustments are not switches.\n"
        "4. Be conservative. If the object identity is ambiguous, keep the segment continuous.\n\n"
        + _strict_json_suffix()
        + "\n### Examples\n"
        "{\n"
        "  \"thought\": \"Images 0-5: robot places a fork. Image 6: hand releases fork and moves to spoon. Image 7: spoon interaction begins.\",\n"
        "  \"transitions\": [6],\n"
        "  \"instructions\": [\"Place the fork\", \"Place the spoon\"]\n"
        "}\n\n"
        "{\n"
        "  \"thought\": \"The robot folds different parts of the same shirt. Object identity does not change.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Fold the shirt\"]\n"
        "}"
    )


def prompt_action_phase_detection(n_images: int) -> str:
    """Fine mode: split by meaningful manipulation phases for InfiData subtasks."""
    return (
        f"You are creating fine-grained InfiData subtask annotations for a {n_images}-frame robot manipulation clip.\n"
        f"Mapping: image indices range from 0 to {n_images - 1}.\n\n"
        "### Goal\n"
        "Detect subtask boundaries at meaningful manipulation phase changes, even when the robot keeps manipulating the same object.\n"
        "Each segment should describe one visually coherent action phase that is useful for robot policy learning.\n\n"
        "### What Counts as a Fine-Grained Subtask Boundary\n"
        "Mark a transition when the dominant manipulation intent visibly changes, such as:\n"
        "1. approach/reach -> grasp/contact\n"
        "2. grasp/contact -> lift/pull/open/spread\n"
        "3. lift/spread -> align/position/place\n"
        "4. align/position -> fold/insert/pour/press/close\n"
        "5. main manipulation -> smooth/adjust/finish/release\n"
        "6. switching to a different object or tool\n\n"
        "### Domain Guidance\n"
        "For laundry or deformable-object tasks, same-object phase changes are important. For example, a shirt-folding episode may include: "
        "reach for shirt, grasp shirt corners, lift or spread shirt, align fabric, fold one side or sleeve, fold the other side, fold the body, smooth or finish.\n"
        "For pick-and-place tasks, useful phases include: reach object, grasp object, lift object, move to target, place/release object.\n"
        "For drawer/container tasks, useful phases include: reach handle, grasp handle, pull/open, interact with contents, push/close.\n\n"
        "### Anti-Oversegmentation Rules\n"
        "Do not mark a boundary for tiny hand jitter, camera shake, brief occlusion, or repeated micro-adjustments with the same intent.\n"
        "Avoid creating many nearly identical segments. Prefer 2-6 segments for a normal short episode unless the video clearly contains more phases.\n"
        "Only mark a transition when the new phase is visually supported by the sampled images.\n"
        "If the whole clip truly shows one continuous phase, return no transitions.\n\n"
        "### Instruction Style\n"
        "Instructions should be short, concrete verb phrases in English, such as \"Reach for the shirt\", \"Grasp the shirt corners\", "
        "\"Fold the left side of the shirt\", or \"Place the block in the bowl\".\n"
        "Use the visible object name when possible. Do not mention frame numbers in instructions.\n\n"
        + _strict_json_suffix()
        + "\n### Examples\n"
        "{\n"
        "  \"thought\": \"Images 0-3: robot reaches toward the shirt. Images 4-6: it grasps shirt corners. Images 7-10: it lifts and spreads the shirt. Images 11-15: it folds the shirt body.\",\n"
        "  \"transitions\": [4, 7, 11],\n"
        "  \"instructions\": [\"Reach for the shirt\", \"Grasp the shirt corners\", \"Lift and spread the shirt\", \"Fold the shirt body\"]\n"
        "}\n\n"
        "{\n"
        "  \"thought\": \"Images 0-4: robot reaches for the block. Images 5-7: it grasps and lifts the block. Images 8-13: it moves the block to the bowl. Images 14-15: it releases the block into the bowl.\",\n"
        "  \"transitions\": [5, 8, 14],\n"
        "  \"instructions\": [\"Reach for the block\", \"Grasp and lift the block\", \"Move the block to the bowl\", \"Release the block into the bowl\"]\n"
        "}\n\n"
        "{\n"
        "  \"thought\": \"All images show the robot continuously smoothing the same cloth with the same intent. No stable phase change is visible.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Smooth the cloth\"]\n"
        "}"
    )
