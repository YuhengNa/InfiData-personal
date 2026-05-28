def prompt_switch_detection(n_images: int, mode: str = "coarse") -> str:
    """Build the VLM prompt for the selected segmentation mode."""
    mode = (mode or "coarse").lower()
    if mode in {"fine", "action_phase"}:
        return prompt_action_phase_detection(n_images)
    return prompt_object_switch_detection(n_images)


def prompt_for_annotations(n_images: int, segmentation_mode: str = "coarse", targets=None) -> str:
    targets = [str(t).lower() for t in (targets or ["subtask"])]
    wants_subtask = "subtask" in targets
    wants_memory = "memory" in targets

    if wants_subtask and wants_memory:
        return prompt_combined_annotation(n_images, segmentation_mode)
    if wants_memory:
        return prompt_memory_summary_detection(n_images)
    return prompt_switch_detection(n_images, mode=segmentation_mode)


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


def _memory_json_suffix(nested: bool = False) -> str:
    prefix = "The memory object" if nested else "The JSON object"
    return (
        "### Output Format: Strict JSON\n"
        "Return only valid JSON. Do not include markdown fences.\n"
        f"{prefix} must contain:\n"
        "- \"thought\": concise reasoning about persistent facts and memory changes.\n"
        "- \"transitions\": a sorted list of image indices where a new memory segment starts.\n"
        "- \"summaries\": one memory summary for each segment. The number of summaries must equal len(transitions) + 1.\n"
        "- \"change_event_types\": one list of event tags for each segment.\n"
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


def prompt_memory_summary_detection(n_images: int) -> str:
    """Memory mode: split by long-term memory state changes."""
    return (
        f"You are creating long-term memory annotations for a {n_images}-frame robot manipulation clip.\n"
        f"Mapping: image indices range from 0 to {n_images - 1}.\n\n"
        "### Goal\n"
        "Detect memory segment boundaries. A memory segment is a time interval where the policy should hold the same long-term memory summary.\n"
        "The summary should contain persistent facts useful for future action, not a frame-by-frame caption.\n\n"
        "### What Long-Term Memory Should Track\n"
        "Write only facts that affect future decisions:\n"
        "1. task progress and completed steps\n"
        "2. hidden or occluded object locations\n"
        "3. container, drawer, door, or tool states\n"
        "4. counts and remaining objects\n"
        "5. failed attempts and recovery needs\n"
        "6. immediate next subgoal when useful\n\n"
        "### When to Start a New Memory Segment\n"
        "Mark a transition when a persistent fact becomes true, becomes false, or must be updated:\n"
        "- an object is picked up, released, placed, inserted, removed, hidden, or revealed\n"
        "- a drawer/container/door is opened or closed\n"
        "- a subtask is completed and the next subgoal changes\n"
        "- a count changes, such as one more block placed\n"
        "- a grasp or attempt fails and the recovery strategy changes\n"
        "Do not create a new memory segment for tiny motion, camera changes, or repeated adjustments when the memory summary remains the same.\n\n"
        "### Summary Style\n"
        "Each summary must be 1-4 short English sentences, ideally 10-60 words.\n"
        "Use natural language only. Do not use labels like Progress:, Facts:, or Issues:.\n"
        "Do not leak future information that is not true yet within that segment.\n"
        "Prefer stable object names and spatial names, such as left drawer, red block, blue bowl.\n\n"
        "If existing subtask annotations are provided, use them as reliable context for what is happening in the window. "
        "Do not answer that no visual information is available unless every provided image is actually blank or unreadable. "
        "When the exact visual state is uncertain, write a conservative memory summary grounded in the task and subtask context.\n\n"
        "### Event Tags\n"
        "Use short snake_case tags such as initial_observation, object_location_seen, object_picked, object_placed, drawer_opened, drawer_closed, "
        "container_opened, count_updated, subtask_completed, grasp_failed, recovery_needed, memory_unchanged.\n\n"
        + _memory_json_suffix()
        + "\n### Examples\n"
        "{\n"
        "  \"thought\": \"Images 0-3 show the red block inside the closed left drawer. Images 4-8 show the drawer being opened. Images 9-15 show the robot reaching toward the block inside the drawer.\",\n"
        "  \"transitions\": [4, 9],\n"
        "  \"summaries\": [\n"
        "    \"The red block is in the left drawer. The drawer is closed. Next, open the left drawer.\",\n"
        "    \"The red block is in the left drawer. The drawer is open. Next, reach into the left drawer.\",\n"
        "    \"The red block is in the open left drawer. Next, grasp the red block.\"\n"
        "  ],\n"
        "  \"change_event_types\": [[\"initial_observation\", \"object_location_seen\"], [\"drawer_opened\"], [\"subtask_completed\"]]\n"
        "}\n\n"
        "{\n"
        "  \"thought\": \"The robot continuously moves the block toward the bowl, but no persistent memory fact changes inside this sampled window.\",\n"
        "  \"transitions\": [],\n"
        "  \"summaries\": [\"The robot is holding the block. The bowl is the target location. Next, move the block to the bowl.\"],\n"
        "  \"change_event_types\": [[\"memory_unchanged\"]]\n"
        "}"
    )


def prompt_combined_annotation(n_images: int, segmentation_mode: str = "coarse") -> str:
    subtask_prompt = prompt_switch_detection(n_images, mode=segmentation_mode)
    memory_prompt = prompt_memory_summary_detection(n_images)
    return (
        f"You are annotating a {n_images}-frame robot manipulation clip for InfiData.\n"
        f"Mapping: image indices range from 0 to {n_images - 1}.\n\n"
        "Return both subtask segmentation and long-term memory segmentation in one strict JSON object.\n"
        "Top-level keys must be \"subtask\" and \"memory\".\n\n"
        "### Subtask Annotation Instructions\n"
        + subtask_prompt
        + "\n\n### Memory Annotation Instructions\n"
        + memory_prompt
        + "\n\n### Combined Output Format\n"
        "{\n"
        "  \"subtask\": {\n"
        "    \"thought\": \"...\",\n"
        "    \"transitions\": [4, 9],\n"
        "    \"instructions\": [\"Reach for the shirt\", \"Grasp the shirt\", \"Fold the shirt\"]\n"
        "  },\n"
        "  \"memory\": {\n"
        "    \"thought\": \"...\",\n"
        "    \"transitions\": [9],\n"
        "    \"summaries\": [\"The shirt is on the table. Next, grasp it.\", \"The robot is holding the shirt. Next, fold it.\"],\n"
        "    \"change_event_types\": [[\"initial_observation\"], [\"object_picked\", \"subtask_completed\"]]\n"
        "  }\n"
        "}"
    )
