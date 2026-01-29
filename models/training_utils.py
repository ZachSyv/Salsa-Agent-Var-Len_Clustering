import torch
from torch.nn.utils import rnn

# Training utils
def build_one_instance(tokenizer, captions, motion):
    input_ids, target_ids = [], []
    bos = tokenizer.bos_token_id
    input_ids.append(bos)
    target_ids.append(-100)  # do not perform loss regression on human prompt
    texts = ''
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    input_text = '### Input:\n' + captions + '\n\nResponse: <Motion>'
    text = prompt + instruction + input_text
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
    # So far, the target_ids is equal to input_ids all -100
    text = '</Motion><eos>'
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    # print(one_input_id)
    # print(one_input_id)
    # print(motion)
    input_ids += motion.tolist() + one_input_id  # so the input and output should be equal, except for the human prompt.
    target_ids += motion.tolist() + one_input_id
    return input_ids, target_ids

def process_batch(tokenizer, batch_of_captions, max_tgt_len, batch_of_motions,
                  batch_of_motionscript, batch_of_audio):
    batch_input_ids, batch_target_ids = [], []
    for caption, motion, ms_segments, audio in zip(batch_of_captions, batch_of_motions,
                               batch_of_motionscript, batch_of_audio):

        # one_input_ids, one_target_ids = build_one_instance(tokenizer, caption, motion)
        one_input_ids, one_target_ids = build_training_instance_salsa(tokenizer=tokenizer,
                                                                caption=caption,
                                                                motion_tokens=motion,
                                                                motion_script_segments=ms_segments)
                                                                # audio)
        # build_random_training_instance_salsa(
        #     tokenizer,
        #     leader_motion_script_segments,
        #     follower_motion_script_segments,
        #     leader_motion_tokens,
        #     follower_motion_tokens,
        #     audio_tokens,
        #     proficiency_level,
        #     allowed_tasks="caption_script_to_motion",
        #     snippet_prob=0.3,
        #     min_snippet_steps=1,
        #     max_snippet_steps=4,
        # )

        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def build_training_instance_salsa(tokenizer, caption, motion_tokens, motion_script_segments):
    """
    Build a single training instance:
    Input: coarse caption + motion script (given as a list of segments)
    Output: motion token sequence
    """

    input_ids = []
    target_ids = []
    bos_token_id = tokenizer.bos_token_id

    # Start with BOS token
    input_ids.append(bos_token_id)
    target_ids.append(-100)  # Ignore BOS during loss calculation

    # Prepare motion script (join list with <SEP>)
    joined_motion_script = format_motionscript_bins(motion_script_segments)

    # Build the full input prompt
    system_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following human motion description and detailed motion script.\n\n"
    input_section = f"### Input:\n{caption}\n\n"
    motion_script_section = f"### MotionScript:\n{joined_motion_script}\n\n"
    response_marker = "Response: <Motion>"

    full_prompt = system_prompt + instruction + input_section + motion_script_section + response_marker

    # Tokenize the input (prompt part)
    prompt_token_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    input_ids.extend(prompt_token_ids)
    target_ids.extend([-100] * len(prompt_token_ids))  # Mask prompt part for loss

    # Tokenize motion closing
    motion_end_token_ids = tokenizer("</Motion><eos>", add_special_tokens=False).input_ids

    # Append the motion tokens and motion end marker
    input_ids.extend(motion_tokens.tolist())
    input_ids.extend(motion_end_token_ids)

    target_ids.extend(motion_tokens.tolist()) # We already tokenized motion_tokens into LLMs tokenizer
    target_ids.extend(motion_end_token_ids)

    return input_ids, target_ids

def format_motionscript_bins(motion_script_segments, window_sec=0.5):
    """
    Prepare motion script text from list of motion descriptions.

    Args:
        motion_script_segments: List[str], each item covering `window_sec` seconds
        window_sec: float, duration each segment covers (default 0.5s)

    Returns:
        A single formatted string ready to insert into training prompt
    """
    script_lines = []
    current_time = 0.0

    for description in motion_script_segments:
        start = current_time
        end = current_time + window_sec

        if description.strip() == "":
            description_text = "<Motionless>"
        else:
            description_text = description.strip()

        line = f"{start:.1f}s-{end:.1f}s: {description_text}"
        script_lines.append(line)

        current_time = end

    # Join all segments with <SEP> separator
    joined_script = " <SEP> ".join(script_lines)
    return joined_script


SINGLE_DANCER_CAPTIONS = {
    "beginner": [
        "A beginner salsa dancer practices simple steps with careful timing.",
        "A novice salsa dancer moves cautiously to the rhythm.",
        "A beginner salsa dancer performs basic footwork with focused effort.",
        "A new salsa dancer follows the beat with steady and controlled movements.",
        "A first-time salsa dancer attempts a slow and structured routine."
    ],
    "intermediate": [
        "An intermediate salsa dancer combines footwork and turns with growing confidence.",
        "A mid-level salsa dancer executes a balanced and expressive routine.",
        "An intermediate dancer performs with more rhythm and body coordination.",
        "A salsa dancer at intermediate level adds flair while maintaining structure.",
        "An intermediate-level salsa dancer blends technical steps with smoother transitions."
    ],
    "professional": [
        "A professional salsa dancer delivers a dynamic and polished performance.",
        "A skilled salsa dancer flows through complex moves with ease.",
        "A professional dancer commands the floor with sharp and expressive motion.",
        "A seasoned salsa dancer performs an intricate routine with confidence.",
        "An expert salsa dancer dazzles with swift, precise, and rhythmic movements."
    ]
}
TWO_DANCER_CAPTIONS = {
    "beginner": [
        "Two beginner salsa dancers move cautiously together, focusing on basic steps.",
        "A novice salsa couple performs a simple, synchronized routine with steady rhythm.",
        "New salsa dance partners coordinate basic footwork with careful timing.",
        "First-time salsa dancers attempt a structured and slow partner routine.",
        "Beginner salsa partners follow the beat closely with controlled movements."
    ],
    "intermediate": [
        "An intermediate salsa couple performs turns and cross-body leads with growing fluidity.",
        "Two intermediate dancers showcase balanced footwork and expressive partnerwork.",
        "Intermediate salsa partners coordinate steps and spins with increasing confidence.",
        "A salsa duo at intermediate level blends technical movements with smoother transitions.",
        "Intermediate-level salsa dancers add flair while maintaining clear structure together."
    ],
    "professional": [
        "A professional salsa couple dazzles with sharp, synchronized, and rhythmic movements.",
        "Expert salsa dance partners flow through complex figures with precision and style.",
        "Two skilled dancers deliver a dynamic and polished partner performance.",
        "A seasoned salsa duo commands the floor with intricate and expressive choreography.",
        "Professional salsa partners showcase effortless timing and seamless coordination."
    ]
}

def generate_coarse_caption(proficiency, two_dancers=False, role=None):
    """
    Generate a coarse caption based on proficiency and task type.
    If single dancer, also inject 'leader' or 'follower' into the caption.
    """

    if two_dancers:
        caption = random.choice(TWO_DANCER_CAPTIONS[proficiency])
    else:
        assert role in ["leader", "follower"], "Role must be specified for single dancer tasks"
        base_caption = random.choice(SINGLE_DANCER_CAPTIONS[proficiency])
        # Replace 'dancer' with 'leader' or 'follower'
        if "dancer" in base_caption:
            caption = base_caption.replace("dancer", role)
        else:
            # Just in case (shouldn't happen), fallback
            caption = base_caption
    return caption

def build_training_instance_salsa(
    tokenizer,
    caption,
    motion_tokens,
    motion_script_segments,
    audio_tokens=None  # New optional input
):
    """
    Build a single training instance:
    Input: coarse caption + motion script (+ optional audio tokens)
    Output: motion token sequence
    """

    input_ids = []
    target_ids = []
    bos_token_id = tokenizer.bos_token_id

    # Start with BOS token
    input_ids.append(bos_token_id)
    target_ids.append(-100)  # Ignore BOS during loss calculation

    # Prepare motion script (join list with <SEP>)
    joined_motion_script = format_motionscript_bins(motion_script_segments)

    # --- Build prompt sections ---
    system_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following human motion description and detailed motion script.\n\n"
    input_section = f"### Input:\n{caption}\n\n"
    motion_script_section = f"### MotionScript:\n{joined_motion_script}\n\n"

    # Optionally add audio section if available
    if audio_tokens is not None:
        audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n"
    else:
        audio_section = ""

    response_marker = "Response: <Motion>"

    # --- Build full prompt ---
    full_prompt = system_prompt + instruction + input_section + motion_script_section + audio_section + response_marker

    # Tokenize the prompt part
    prompt_token_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    input_ids.extend(prompt_token_ids)
    target_ids.extend([-100] * len(prompt_token_ids))  # Mask prompt part for loss

    # Tokenize motion closing
    motion_end_token_ids = tokenizer("</Motion><eos>", add_special_tokens=False).input_ids

    # Append the motion tokens and motion end marker
    input_ids.extend(motion_tokens.tolist())
    input_ids.extend(motion_end_token_ids)

    target_ids.extend(motion_tokens.tolist())
    target_ids.extend(motion_end_token_ids)

    return input_ids, target_ids




'''
def build_random_training_instance_salsa_old(
    tokenizer,
    leader_motion_script_segments,
    follower_motion_script_segments,
    leader_motion_tokens,
    follower_motion_tokens,
    audio_tokens,
    proficiency_level,
    allowed_tasks=None,
    snippet_prob=0.3,
    min_snippet_steps=1,
    max_snippet_steps=4,
):
    
    # Build a full random training instance (motion/script/audio/leader/follower), ready for batching.
    

    # 1. Pick task
    all_tasks = [
        "caption_script_to_motion",
        "caption_script_audio_to_motion",
        "leader_to_follower",
        "follower_to_leader",
        "caption_to_motionscript",
        "caption_to_motionscript_audio",
        "motionscript_to_motion",
        "motion_to_motionscript",
        "snippet_script_to_motion",
        "snippet_motion_to_script",
    ]
    task = random.choice(allowed_tasks if allowed_tasks else all_tasks)

    # 2. Random snippet decision
    snippet_task = False
    if task.startswith("snippet_") or random.random() < snippet_prob:
        snippet_task = True

    # 3. Pick input role
    if task in ["caption_script_to_motion", "caption_script_audio_to_motion", "caption_to_motionscript", "caption_to_motionscript_audio", "motionscript_to_motion"]:
        role = random.choice(["leader", "follower"])
    elif task == "leader_to_follower":
        role = "leader"
    elif task == "follower_to_leader":
        role = "follower"
    else:
        role = None  # two-person tasks

    # 4. Pick caption
    if role:
        caption = generate_coarse_caption(proficiency_level, two_dancers=False, role=role)
    else:
        caption = generate_coarse_caption(proficiency_level, two_dancers=True)

    # 5. Pick data for leader/follower
    if role == "leader":
        motion_script_segments = leader_motion_script_segments
        motion_tokens = leader_motion_tokens
    elif role == "follower":
        motion_script_segments = follower_motion_script_segments
        motion_tokens = follower_motion_tokens
    else:
        motion_script_segments = None  # two-person
        motion_tokens = None

    # 6. Snippet slicing
    if snippet_task and motion_script_segments is not None:
        snippet_size = random.randint(min_snippet_steps, max_snippet_steps)
        start_idx = random.randint(0, max(0, len(motion_script_segments) - snippet_size))

        motion_script_segments = motion_script_segments[start_idx:start_idx + snippet_size]
        motion_tokens = motion_tokens[start_idx * 2: (start_idx + snippet_size) * 2]
        audio_tokens = audio_tokens[start_idx * 20: (start_idx + snippet_size) * 20]

    # 7. Audio random inclusion
    allow_audio = task in [
        "caption_script_audio_to_motion",
        "caption_to_motionscript_audio", # X we never want to predict audio why you want to do this?
        "leader_to_follower",
        "follower_to_leader",
        "motionscript_to_motion",
        "snippet_script_to_motion"
    ]
    use_audio = (audio_tokens is not None) and allow_audio and (random.random() < 0.5)

    # 8. Initialize
    input_ids = []
    target_ids = []
    bos_token_id = tokenizer.bos_token_id

    input_ids.append(bos_token_id)
    target_ids.append(-100)

    system_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )

    input_section, motion_script_section, audio_section, response_marker = "", "", "", ""

    # 9. Build sections
    if task == "caption_script_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence based on description and motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = f"### MotionScript:\n<{role.capitalize()}Script> {format_motionscript_bins(motion_script_segments)} </{role.capitalize()}Script>\n\n"
        response_marker = "<Motion>" # X what we want to do, we want to use general motion token or folower/leader?
        target_tokens = motion_tokens

    elif task == "caption_script_audio_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence based on description, motion script, and music.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = f"### MotionScript:\n<{role.capitalize()}Script> {format_motionscript_bins(motion_script_segments)} </{role.capitalize()}Script>\n\n"
        audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n" if use_audio else ""
        response_marker = "<Motion>"
        target_tokens = motion_tokens

    elif task == "leader_to_follower":
        # X is this okay to use and optionally music or we should add music to the instruction when ever it is available
        instruction = "### Instruction:\nGiven leader motion (and optionally music), predict follower motion.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = "<LeaderMotion> " + " ".join(map(str, leader_motion_tokens)) + " </LeaderMotion>\n\n"
        if use_audio:
            audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n"
        response_marker = "<FollowerMotion>"
        target_tokens = follower_motion_tokens

    elif task == "follower_to_leader":
        instruction = "### Instruction:\nGiven follower motion (and optionally music), predict leader motion.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = "<FollowerMotion> " + " ".join(map(str, follower_motion_tokens)) + " </FollowerMotion>\n\n"
        if use_audio:
            audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n"
        response_marker = "<LeaderMotion>"
        target_tokens = leader_motion_tokens
    # X when using follower to leader, do wee need to incorporate motion scrpts somehow
    # for example follower's motionscript to leader's motion script, then motionscripts of leader to motion?
    elif task == "caption_to_motionscript":
        instruction = "### Instruction:\nGenerate a detailed motion script from the description.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        response_marker = f"<{role.capitalize()}Script>"
        target_tokens = tokenizer(format_motionscript_bins(motion_script_segments), add_special_tokens=False).input_ids

    elif task == "caption_to_motionscript_audio": # X naming doesn't look right, this is e.g., caption+audio to motionscript
        instruction = "### Instruction:\nGenerate a detailed motion script from description and music.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n" if use_audio else ""
        response_marker = f"<{role.capitalize()}Script>"
        target_tokens = tokenizer(format_motionscript_bins(motion_script_segments), add_special_tokens=False).input_ids

    elif task == "motionscript_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence from the provided motion script.\n\n"
        input_section = ""
        motion_script_section = f"<{role.capitalize()}Script> {format_motionscript_bins(motion_script_segments)} </{role.capitalize()}Script>\n\n"
        audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n" if use_audio else ""
        response_marker = "<Motion>"
        target_tokens = motion_tokens

    elif task == "motion_to_motionscript":
        instruction = "### Instruction:\nDescribe the following motion sequence in a detailed motion script.\n\n"
        # Randomly one person or both
        if random.random() < 0.5:
            # Single person
            if random.random() < 0.5:
                motion_script_section = "<LeaderMotion> " + " ".join(map(str, leader_motion_tokens)) + " </LeaderMotion>\n\n"
                response_marker = "<LeaderScript>"
                target_tokens = tokenizer(format_motionscript_bins(leader_motion_script_segments), add_special_tokens=False).input_ids
            else:
                motion_script_section = "<FollowerMotion> " + " ".join(map(str, follower_motion_tokens)) + " </FollowerMotion>\n\n"
                response_marker = "<FollowerScript>" # X from the original code I remember they were using sth like '### Input:\n' + captions + '\n\nResponse: <Motion>'
                # so, do we also need to add that.
                target_tokens = tokenizer(format_motionscript_bins(follower_motion_script_segments), add_special_tokens=False).input_ids
        else:
            # Both dancers
            motion_script_section = (
                    "<LeaderMotion> " + " ".join(map(str, leader_motion_tokens)) + " </LeaderMotion>\n\n"
                                                                                   "<FollowerMotion> " + " ".join(
                map(str, follower_motion_tokens)) + " </FollowerMotion>\n\n"
            )
            # X what is happening with that map str bloa bla... we didn't have something like that in the original starting code don't forget motion tokens are already in the LLM tokenizer space
            response_marker = ""  # no need here, because response is both parts X I already explained above.
            target_text = (
                    "<LeaderScript> " + format_motionscript_bins(leader_motion_script_segments) + " </LeaderScript>\n"
                                                                                                  "<FollowerScript> " + format_motionscript_bins(
                follower_motion_script_segments) + " </FollowerScript>"
            )

            target_tokens = tokenizer(target_text, add_special_tokens=False).input_ids
            # X generally this section seems incorrect. I don't know why, please make sure.
    elif task == "snippet_script_to_motion":
        instruction = "### Instruction:\nGenerate a short motion snippet from the partial motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = f"<{role.capitalize()}Script> {format_motionscript_bins(motion_script_segments)} </{role.capitalize()}Script>\n\n"
        audio_section = "<AudioTokens> " + " ".join(map(str, audio_tokens)) + " </AudioTokens>\n\n" if use_audio else ""
        response_marker = "<Motion>" # X again
        target_tokens = motion_tokens

    elif task == "snippet_motion_to_script":
        instruction = "### Instruction:\nDescribe the following motion snippet in short motion script.\n\n"
        motion_script_section = "<Motion> " + " ".join(map(str, motion_tokens)) + " </Motion>\n\n"
        response_marker = "<MotionScript>"
        target_tokens = tokenizer(format_motionscript_bins(motion_script_segments), add_special_tokens=False).input_ids

    else:
        raise ValueError(f"Unknown task type: {task}")

    # 10. Assemble full prompt
    full_prompt = (
        system_prompt
        + instruction
        + input_section
        + motion_script_section
        + audio_section
        + f"Response: {response_marker}"
    )

    prompt_token_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    input_ids.extend(prompt_token_ids)
    target_ids.extend([-100] * len(prompt_token_ids))

    input_ids.extend(target_tokens)
    target_ids.extend(target_tokens)

    # 11. Close
    if response_marker.endswith("<Motion>") or response_marker.endswith("<LeaderMotion>") or response_marker.endswith("<FollowerMotion>"):
        end_token_ids = tokenizer("</Motion><eos>", add_special_tokens=False).input_ids
    elif response_marker.endswith("<MotionScript>") or response_marker.endswith("<LeaderScript>") or response_marker.endswith("<FollowerScript>"):
        end_token_ids = tokenizer("</MotionScript><eos>", add_special_tokens=False).input_ids
    else:
        end_token_ids = []

    input_ids.extend(end_token_ids)
    target_ids.extend(end_token_ids)

    return input_ids, target_ids, task
'''

# Full final version coming up next



# Full final version of build_random_training_instance_salsa with all original, cross-role, and combined tasks


import random
PAIR2LEVEL = {
    f"pair{i}": level
    for i, level in zip(range(1, 10), ["beginner", "intermediate", "professional"] * 3)
}
def process_batch_Salsa(tokenizer, batch_aux_info, batch_ms_desc_L, batch_ms_des_F,
                                    batch_vq_tokens_L, batch_vq_tokens_F,
                                    batch_audio_tokens,
                                    max_tgt_len, current_batch_task=None,
                                    motion_repr_type='humanml3d',
                                    batch_interhuman_data=None,
                                    include_audio=False):

    batch_input_ids, batch_target_ids = [], []
    use_interhuman = (motion_repr_type == 'interhuman') and (batch_interhuman_data is not None)
    if not current_batch_task:
        current_batch_task = random.choice(all_tasks)

    n_samples = len(batch_aux_info) if hasattr(batch_aux_info, '__len__') else 1
    for i in range(n_samples):
        aux = batch_aux_info[i] if n_samples > 1 else batch_aux_info
        ms_desc_L = batch_ms_desc_L[i] if n_samples > 1 else batch_ms_desc_L
        ms_des_F = batch_ms_des_F[i] if n_samples > 1 else batch_ms_des_F
        vq_tokens_L = batch_vq_tokens_L[i] if n_samples > 1 else batch_vq_tokens_L
        vq_tokens_F = batch_vq_tokens_F[i] if n_samples > 1 else batch_vq_tokens_F
        audio_tokens = batch_audio_tokens[i] if n_samples > 1 else batch_audio_tokens
        level = aux

        ih_elem = (batch_interhuman_data[i] if n_samples > 1 else (batch_interhuman_data[0] if isinstance(batch_interhuman_data, (list, tuple)) else batch_interhuman_data)) if batch_interhuman_data is not None else None
        if use_interhuman and ih_elem is not None:
            ih = ih_elem
            leader_tokens = torch.as_tensor(ih['leader_tokens']).cpu().ravel().tolist()
            follower_tokens = torch.as_tensor(ih['follower_tokens']).cpu().ravel().tolist()
            relationship_tokens = torch.as_tensor(ih['relationship_tokens']).cpu().ravel().tolist()
            aud_list = torch.as_tensor(audio_tokens).cpu().ravel().tolist() if audio_tokens is not None else None
            
            # Use current_batch_task if it's an InterHuman task, otherwise random choice
            if current_batch_task and current_batch_task in INTERHUMAN_TASKS:
                task = current_batch_task
            else:
                task = random.choice(INTERHUMAN_TASKS)
            
            # Extract move_annotations from aux_info if available
            move_annotations = None
            if isinstance(aux, dict) and 'dance_moves' in aux:
                move_annotations = aux['dance_moves']
            
            # Use caption from aux_info if available, otherwise None
            caption = None
            if isinstance(aux, dict) and 'caption' in aux:
                caption = aux['caption']
            
            # Use include_audio parameter (from args) instead of random
            use_audio = include_audio and (aud_list is not None and len(aud_list) > 0)
            
            prompt_text, target_text = build_prompt_interhuman_salsa(
                leader_tokens=leader_tokens,
                follower_tokens=follower_tokens,
                relationship_tokens=relationship_tokens,
                task=task,
                move_annotations=move_annotations,
                level=level if isinstance(level, str) else None,
                caption=caption,
                audio_tokens=aud_list,
                include_audio=use_audio,
            )
            one_input_ids, one_target_ids = interhuman_prompt_target_to_ids(tokenizer, prompt_text, target_text)
        else:
            one_input_ids, one_target_ids, task = build_random_training_instance_salsa_prompt(
                tokenizer=tokenizer,
                leader_motion_script_segments=ms_desc_L.split('-->') if isinstance(ms_desc_L, str) else [],
                follower_motion_script_segments=ms_des_F.split('-->') if isinstance(ms_des_F, str) else [],
                leader_motion_tokens=vq_tokens_L,
                follower_motion_tokens=vq_tokens_F,
                audio_tokens=audio_tokens,
                proficiency_level=level,
                allowed_tasks=[current_batch_task],
                snippet_prob=0.5,
                min_snippet_steps=1,
                max_snippet_steps=4,
            )

        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()

    # with open('batch_max_tokens.txt', 'a') as f:
    #     f.write(f'{input_ids.size()}\n')
    # assert input_ids.shape[1] < max_tgt_len

    # with open('myfile.txt', 'a') as f:
    #     f.write('Hello, world!\n')
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

all_tasks = [
        "caption_to_motion",
        "caption_script_to_motion",
        "caption_script_audio_to_motion",
        "leader_to_follower",
        "follower_to_leader",
        "caption_to_motionscript",
        "caption_audio_to_motionscript",
        "motionscript_to_motion",
        "motion_to_motionscript",
        "motion_completion"
    ]

def build_random_training_instance_salsa_prompt(
    tokenizer,
    leader_motion_script_segments,
    follower_motion_script_segments,
    leader_motion_tokens,
    follower_motion_tokens,
    audio_tokens,
    proficiency_level,
    allowed_tasks="caption_script_to_motion",
    snippet_prob=0.3,
    min_snippet_steps=1,
    max_snippet_steps=4,
    inference=False,
    is_baseline=False
):
    #Todo: motion token should be in str format.
    # It means that we need to have something like 'motion_0'



    task = random.choice(allowed_tasks if allowed_tasks else all_tasks)
    snippet_task = task.startswith("snippet_") or random.random() < snippet_prob

    if task in [
        "caption_to_motion",
        "caption_script_to_motion",
        "caption_script_audio_to_motion",
        # "leader_to_follower",
        # "follower_to_leader",
        "caption_to_motionscript",
        "caption_audio_to_motionscript",
        "motionscript_to_motion",
        "motion_to_motionscript",
        "motion_completion"
    ]:
        role = random.choice(["leader", "follower"])
    elif task in ["leader_to_follower", "leader_motion_script_to_follower_script", "leader_motion_script_to_follower_script_motion"]:
        role = "leader"
    elif task in ["follower_to_leader", "follower_motion_script_to_leader_script", "follower_motion_script_to_leader_script_motion"]:
        role = "follower"
    else:
        role = None

    caption = generate_coarse_caption(proficiency_level, two_dancers=(role is None), role=role)

    if role == "leader":
        motion_script_segments = leader_motion_script_segments
        motion_tokens = leader_motion_tokens
    elif role == "follower":
        motion_script_segments = follower_motion_script_segments
        motion_tokens = follower_motion_tokens
    else:
        motion_script_segments = None
        motion_tokens = None


    if snippet_task and motion_script_segments is not None:
        snippet_size = random.randint(min_snippet_steps, max_snippet_steps)
        start_idx = random.randint(0, max(0, len(motion_script_segments) - snippet_size))
        # Sampling rates:
        # motion_script_segments → 2 per second
        # motion_tokens → 5 per second (2.5x faster than motion_script_segments)
        # audio_tokens → 40 per second (20x faster than motion_script_segments)
        motion_script_segments = motion_script_segments[start_idx:start_idx + snippet_size]
        motion_tokens = motion_tokens[start_idx * 5 // 2: (start_idx + snippet_size) * 5 // 2]
        audio_tokens = audio_tokens[start_idx * 20: (start_idx + snippet_size) * 20]

    allow_audio = task in [
        # "caption_to_motion",
        # "caption_script_to_motion",
        "caption_script_audio_to_motion",
        # "leader_to_follower",
        # "follower_to_leader",
        # "caption_to_motionscript",
        "caption_audio_to_motionscript",
        # "motionscript_to_motion",
        # "motion_to_motionscript",
        # "motion_completion"
    ]
    use_audio = (audio_tokens is not None) and allow_audio and (random.random() < 0.5)


    system_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )

    def wrap_script(role, script):
        return f"### Script:\n<{role.capitalize()}Script> {format_motionscript_bins(script)} </{role.capitalize()}Script>"
    def wrap_audio(audio_tokenz):
        return "### Audio:\n<AudioTokens> " + \
                            ''.join([ f'<Audio_{tok}>' for tok in audio_tokenz]) + \
                            " </AudioTokens>\n\n"




    if task == "motion_completion":
        instruction = "### Instruction:\nGiven a partial motion sequence, complete the motion.\n\n"
        input_section = f"### Input:\n{caption}\n\n"

        # Let's slice first N tokens (e.g., first 30%)
        split_ratio = random.uniform(0.2, 0.5)
        split_idx = int(len(motion_tokens) * split_ratio)
        partial_motion = f"<{role.capitalize()}Motion>" + ''.join(
            [f'<Motion_{tok}>' for tok in motion_tokens[:split_idx]]) + f"</{role.capitalize()}Motion>  \n\n"

        response_text = f"Response: <{role.capitalize()}Motion>"

        section_prompt = instruction + input_section + partial_motion + response_text

        target_text = ''.join(
            [f'<Motion_{tok}>' for tok in motion_tokens[split_idx:]]) + f"</{role.capitalize()}Motion>"

    elif task == "caption_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence based on the description.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        response_text = f"Response: <{role.capitalize()}Motion>"
        # it was for when I was using motion_tokens in LLM's tokenizer space
        # target_tokens = motion_tokens
        # Now we have the motion tokens in the vq tokenizer space, so:
        # We need to join without space in order to avoid '_' tokens
        section_prompt = instruction + input_section  + response_text
        target_text = ''.join([f'<Motion_{tok}>' for tok in motion_tokens])
        target_text += (f"</{role.capitalize()}Motion>")

    if task == "caption_script_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence based on the description and motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = wrap_script(role, motion_script_segments) + "\n\n"
        response_text = f"Response: <{role.capitalize()}Motion>"
        section_prompt = instruction + input_section + motion_script_section  + response_text

        target_text = ''.join([f'<Motion_{tok}>' for tok in motion_tokens])
        target_text += (f"</{role.capitalize()}Motion>")

    elif task == "caption_script_audio_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence based on description and motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = wrap_script(role, motion_script_segments) + "\n\n"
        audio_section = ''
        if use_audio:
            instruction = "### Instruction:\nGenerate a motion sequence based on description, motion script, and music.\n\n"
            audio_section = wrap_audio(audio_tokens)
        response_text = f"Response: <{role.capitalize()}Motion>"

        section_prompt = instruction + input_section + motion_script_section + audio_section + response_text

        target_text = ''.join([ f'<Motion_{tok}>' for tok in motion_tokens])
        target_text += (f"</{role.capitalize()}Motion>")


    elif task == "leader_to_follower":
        instruction = "### Instruction:\nGiven leader motion, predict follower motion.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        # motion_script_section = wrap_motion("leader", leader_motion_tokens) + "\n\n"
        leader_motion = "<LeaderMotion>" + \
                        ''.join([ f'<Motion_{tok}>' for tok in leader_motion_tokens]) + \
                        "</LeaderMotion>"
        audio_section = ''
        if use_audio:
            instruction = "### Instruction:\nGiven leader motion, and music, predict follower motion.\n\n"
            audio_section = wrap_audio(audio_tokens)
        response_text = "Response: <FollowerMotion>"

        section_prompt = instruction + input_section + leader_motion + audio_section + response_text

        target_text = ''.join([f'<Motion_{tok}>' for tok in follower_motion_tokens])
        target_text += (f"</FollowerMotion>")

    elif task == "follower_to_leader":
        instruction = "### Instruction:\nGiven follower motion, predict leader motion.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        # motion_script_section = wrap_motion("leader", leader_motion_tokens) + "\n\n"
        follower_motion = "<FollowerMotion>" + \
                        ''.join([f'<Motion_{tok}>' for tok in follower_motion_tokens]) + \
                        "</FollowerMotion>"
        audio_section = ''
        if use_audio:
            instruction = "### Instruction:\nGiven follower motion, and music, predict leader motion.\n\n"
            audio_section = wrap_audio(audio_tokens)
        response_text = "Response: <LeaderMotion>"

        section_prompt = instruction + input_section + follower_motion + audio_section + response_text

        target_text = ''.join([f'<Motion_{tok}>' for tok in leader_motion_tokens])
        target_text += (f"</LeaderMotion>")

    elif task == "caption_to_motionscript":
        instruction = "### Instruction:\nGenerate a detailed motion script based on the description.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        response_text = f"Response: <{role.capitalize()}Script>"

        section_prompt = instruction + input_section + response_text
        target_text = format_motionscript_bins(motion_script_segments) + f" </{role.capitalize()}Script>"

    elif task == "caption_audio_to_motionscript":
        instruction = "### Instruction:\nGenerate a detailed motion script based on the description and music.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        audio_section = wrap_audio(audio_tokens)
        response_text = f"Response: <{role.capitalize()}Script>"

        section_prompt = instruction + input_section + audio_section + response_text
        target_text = format_motionscript_bins(motion_script_segments) + f" </{role.capitalize()}Script>"

    elif task == "motionscript_to_motion":
        instruction = "### Instruction:\nGenerate a motion sequence from the provided motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        motion_script_section = wrap_script(role, motion_script_segments) + "\n\n"
        audio_section = ''
        if use_audio:
            instruction = "### Instruction:\nGenerate a motion sequence from the provided motion script and music.\n\n"
            audio_section = wrap_audio(audio_tokens)
        response_text = f"Response: <{role.capitalize()}Motion>"

        section_prompt = instruction + input_section+ motion_script_section + audio_section + response_text
        target_text = ''.join([f'<Motion_{tok}>' for tok in motion_tokens]) + f"</{role.capitalize()}Motion>"


    elif task == "motion_to_motionscript":
        instruction = "### Instruction:\nDescribe the following motion sequence in a detailed motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        ms = f"<{role.capitalize()}Motion>" + \
             ''.join([f'<Motion_{tok}>' for tok in motion_tokens]) + \
             f"</{role.capitalize()}Motion>" + "\n\n"
        response_text = f"Response: <{role.capitalize()}Script>"
        section_prompt = instruction + input_section + ms  + response_text

        motion_script_section = format_motionscript_bins(motion_script_segments) + \
                                f"</{role.capitalize()}Script>"
        target_text = motion_script_section

    elif task == "motion_to_caption":
        instruction = "### Instruction:\nDescribe the following motion sequence in natural language.\n\n"
        motion_section = "<" + role.capitalize() + "Motion>" + \
                         ''.join([f'<Motion_{tok}>' for tok in motion_tokens]) + \
                         f"</{role.capitalize()}Motion>\n\n"
        audio_section = ""
        if use_audio:
            instruction = "### Instruction:\nDescribe the following motion sequence and music in natural language.\n\n"
            audio_section = wrap_audio(audio_tokens)

        response_text = "Response: "
        section_prompt = instruction + motion_section + audio_section + response_text

        target_text = caption  # Directly using caption as target

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    elif task == "leader_motion_script_to_follower_script":
        instruction = "### Instruction:\nGiven leader's motion script and motion, predict follower's motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        leader_script_section = wrap_script("leader", leader_motion_script_segments) + "\n\n"
        leader_motion = "<LeaderMotion>" + \
                        ''.join([f'<Motion_{tok}>' for tok in leader_motion_tokens]) + \
                        "</LeaderMotion>\n\n"
        response_text = "Response: <FollowerScript>"
        section_prompt = instruction + input_section + leader_script_section + leader_motion + response_text

        target_text = format_motionscript_bins(follower_motion_script_segments) + " </FollowerScript>"

    elif task == "follower_motion_script_to_leader_script":
        instruction = "### Instruction:\nGiven follower's motion script and motion, predict leader's motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        follower_script_section = wrap_script("follower", follower_motion_script_segments) + "\n\n"
        follower_motion = "<FollowerMotion>" + \
                          ''.join([f'<Motion_{tok}>' for tok in follower_motion_tokens]) + \
                          "</FollowerMotion>\n\n"
        response_text = "Response: <LeaderScript>"
        section_prompt = instruction + input_section + follower_script_section + follower_motion + response_text

        target_text = format_motionscript_bins(leader_motion_script_segments) + " </LeaderScript>"

    elif task == "leader_script_to_leader_motion":
        instruction = "### Instruction:\nGenerate leader's motion from the given leader's motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        leader_script_section = wrap_script("leader", leader_motion_script_segments) + "\n\n"
        response_text = "Response: <LeaderMotion>"
        section_prompt = instruction + input_section + leader_script_section + response_text

        target_text = ''.join([f'<Motion_{tok}>' for tok in leader_motion_tokens]) + "</LeaderMotion>"

    elif task == "follower_script_to_follower_motion":
        instruction = "### Instruction:\nGenerate follower's motion from the given follower's motion script.\n\n"
        input_section = f"### Input:\n{caption}\n\n"
        follower_script_section = wrap_script("follower", follower_motion_script_segments) + "\n\n"
        response_text = "Response: <FollowerMotion>"
        section_prompt = instruction + input_section + follower_script_section + response_text

        target_text = ''.join([f'<Motion_{tok}>' for tok in follower_motion_tokens]) + "</FollowerMotion>"





    input_ids, target_ids = [], []
    input_ids.append(tokenizer.bos_token_id)
    target_ids.append(-100)


    full_prompt = system_prompt + section_prompt

    if is_baseline:
        full_prompt = full_prompt.replace('FollowerMotion', 'Motion')
        full_prompt = full_prompt.replace('LeaderMotion', 'Motion')

    prompt_token_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    input_ids.extend(prompt_token_ids)

    if inference:
        return full_prompt, input_ids

    target_ids.extend([-100] * len(prompt_token_ids))

    target_tokens = tokenizer(target_text, add_special_tokens=False).input_ids
    input_ids.extend(target_tokens)
    target_ids.extend(target_tokens)

    # Todo: we need to know whether we want to finish with </motion> or </{ROLEm}motion>
    # end_token_ids = tokenizer("</Motion><eos>", add_special_tokens=False).input_ids
    end_token_ids = tokenizer("<eos>", add_special_tokens=False).input_ids
    input_ids.extend(end_token_ids)
    target_ids.extend(end_token_ids)


    return input_ids, target_ids, task


# -----------------------------------------------------------------------------
# InterHuman Salsa prompt builder (pure text; tokenizer/embedding later)
# -----------------------------------------------------------------------------
# Single source of truth for delimiters so we never write incorrect tokens.
INTERHUMAN_PROMPT_DELIMITERS = {
    "leader_motion_open": "<LeaderMotion>",
    "leader_motion_close": "</LeaderMotion>",
    "follower_motion_open": "<FollowerMotion>",
    "follower_motion_close": "</FollowerMotion>",
    "relationship_open": "<Relationship>",
    "relationship_close": "</Relationship>",
    "audio_open": "<AudioTokens>",
    "audio_close": "</AudioTokens>",
    "audio_token": "<Audio_{}>",  # format with int
    "ih_token": "<IH_{}>",   # format with int: IH_0, IH_1, ...
    "rel_token": "<Rel_{}>", # format with int: Rel_0, Rel_1, ...
    "response_header": "### Response:\n",  # then label + open token of response modality
    "label_leader_motion": "Leader motion: ",
    "label_follower_motion": "Follower motion: ",
    "label_relationship": "Relationship: ",
    "label_music": "Music: ",
    "instruction_leader_rel_to_follower": "### Instruction:\nGiven salsa leader motion and relationship, predict salsa follower motion.\n\n",
    "instruction_leader_rel_to_follower_audio": "### Instruction:\nGiven salsa leader motion, relationship, and music, predict salsa follower motion.\n\n",
    "instruction_follower_rel_to_leader": "### Instruction:\nGiven salsa follower motion and relationship, predict salsa leader motion.\n\n",
    "instruction_follower_rel_to_leader_audio": "### Instruction:\nGiven salsa follower motion, relationship, and music, predict salsa leader motion.\n\n",
    "instruction_caption_leader_rel_to_follower": "### Instruction:\nGiven the salsa description, leader motion, and relationship, predict salsa follower motion.\n\n",
    "instruction_caption_leader_rel_to_follower_audio": "### Instruction:\nGiven the salsa description, leader motion, relationship, and music, predict salsa follower motion.\n\n",
    "instruction_pair_to_relationship": "### Instruction:\nGiven salsa leader and follower motion, predict the relationship between them.\n\n",
    "instruction_pair_to_relationship_audio": "### Instruction:\nGiven salsa leader and follower motion and music, predict the relationship between them.\n\n",
    "instruction_caption_follower_rel_to_leader": "### Instruction:\nGiven the salsa description, follower motion, and relationship, predict salsa leader motion.\n\n",
    "instruction_caption_follower_rel_to_leader_audio": "### Instruction:\nGiven the salsa description, follower motion, relationship, and music, predict salsa leader motion.\n\n",
    "instruction_caption_to_leader": "### Instruction:\nGenerate salsa leader motion from the description.\n\n",
    "instruction_caption_to_leader_audio": "### Instruction:\nGenerate salsa leader motion from the description and music.\n\n",
    "instruction_caption_to_follower": "### Instruction:\nGenerate salsa follower motion from the description.\n\n",
    "instruction_caption_to_follower_audio": "### Instruction:\nGenerate salsa follower motion from the description and music.\n\n",
    "instruction_leader_to_follower": "### Instruction:\nGiven salsa leader motion, predict salsa follower motion.\n\n",
    "instruction_leader_to_follower_audio": "### Instruction:\nGiven salsa leader motion and music, predict salsa follower motion.\n\n",
    "instruction_follower_to_leader": "### Instruction:\nGiven salsa follower motion, predict salsa leader motion.\n\n",
    "instruction_follower_to_leader_audio": "### Instruction:\nGiven salsa follower motion and music, predict salsa leader motion.\n\n",
    "instruction_motion_completion_leader": "### Instruction:\nGiven a partial salsa leader motion, complete the motion.\n\n",
    "instruction_motion_completion_leader_audio": "### Instruction:\nGiven a partial salsa leader motion and music, complete the motion.\n\n",
    "instruction_motion_completion_follower": "### Instruction:\nGiven a partial salsa follower motion, complete the motion.\n\n",
    "instruction_motion_completion_follower_audio": "### Instruction:\nGiven a partial salsa follower motion and music, complete the motion.\n\n",
    "input_section": "### Input:\n",
    "moves_section": "### Moves in this window:\n",
    "system_prompt": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    ),
}

# All special delimiter tokens for InterHuman tokenizer extension (from INTERHUMAN_PROMPT_DELIMITERS).
# Used in mllm when motion_repr_type=='interhuman'. Audio modality tokens <Audio_0>.. are added separately
# only when include_audio is True.
INTERHUMAN_SPECIAL_TOKENS = [
    "<LeaderMotion>", "</LeaderMotion>",
    "<FollowerMotion>", "</FollowerMotion>",
    "<Relationship>", "</Relationship>"
]

# Canonical list of InterHuman Salsa tasks (for UI, training, and validation).
# Excludes motionscript-based tasks; all support optional audio via include_audio.
INTERHUMAN_TASKS = [
    "leader_rel_to_follower",
    "follower_rel_to_leader",
    "caption_leader_rel_to_follower",
    "caption_follower_rel_to_leader",
    "pair_to_relationship",
    "caption_to_leader",
    "caption_to_follower",
    "leader_to_follower",
    "follower_to_leader",
    "motion_completion_leader",
    "motion_completion_follower",
]


def interhuman_prompt_target_to_ids(tokenizer, prompt_text, target_text, prepend_bos=True, append_eos=True):
    """Convert (prompt_text, target_text) from build_prompt_interhuman_salsa to input_ids/target_ids.
    Same convention as build_random_training_instance_salsa_prompt: BOS, prompt masked with -100, target + eos with real ids.
    """
    input_ids, target_ids = [], []
    if prepend_bos:
        input_ids.append(tokenizer.bos_token_id)
        target_ids.append(-100)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    input_ids.extend(prompt_ids)
    target_ids.extend([-100] * len(prompt_ids))
    target_ids_raw = tokenizer(target_text, add_special_tokens=False).input_ids
    input_ids.extend(target_ids_raw)
    target_ids.extend(target_ids_raw)
    if append_eos:
        eos_ids = tokenizer("<eos>", add_special_tokens=False).input_ids
        input_ids.extend(eos_ids)
        target_ids.extend(eos_ids)
    return input_ids, target_ids


def _format_move_summary_for_prompt(move_annotations, max_chars=200):
    """Build a short move summary from annotation dicts for use in prompt text."""
    if not move_annotations:
        return ""
    parts = []
    for m in move_annotations:
        mc = m.get("move_class") or "move"
        desc = (m.get("description") or "")[:60]
        if desc:
            parts.append(f"{mc}: {desc}")
        else:
            parts.append(mc)
    s = "; ".join(parts)
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def build_prompt_interhuman_salsa(
    leader_tokens,
    follower_tokens,
    relationship_tokens,
    task="leader_rel_to_follower",
    move_annotations=None,
    level=None,
    caption=None,
    audio_tokens=None,
    include_audio=False,
    completion_split_ratio=0.3,
):
    """
    Build prompt and target as pure text for InterHuman Salsa representation.
    Uses INTERHUMAN_PROMPT_DELIMITERS so delimiters stay consistent everywhere.

    Args:
        leader_tokens: list/int array of InterHuman motion token ids
        follower_tokens: list/int array of InterHuman motion token ids
        relationship_tokens: list/int array of Relationship token ids
        task: one of INTERHUMAN_TASKS
        move_annotations: optional list of dicts with move_class, description (for caption task)
        level: optional str, e.g. "beginner"
        caption: optional str; if None and task is caption_*, built from level + move summary
        audio_tokens: optional list/int array of audio token ids; used when include_audio=True
        include_audio: if True and audio_tokens provided, appends audio block to input and uses _audio instruction
        completion_split_ratio: for motion_completion_* tasks, fraction of tokens used as input (default 0.3)

    Returns:
        (prompt_text, target_text) both strings
    """
    D = INTERHUMAN_PROMPT_DELIMITERS
    leader_tokens = list(leader_tokens) if hasattr(leader_tokens, "__iter__") and not isinstance(leader_tokens, str) else []
    follower_tokens = list(follower_tokens) if hasattr(follower_tokens, "__iter__") and not isinstance(follower_tokens, str) else []
    relationship_tokens = list(relationship_tokens) if hasattr(relationship_tokens, "__iter__") and not isinstance(relationship_tokens, str) else []
    _audio_list = []
    if audio_tokens is not None and hasattr(audio_tokens, "__iter__") and not isinstance(audio_tokens, str):
        _audio_list = list(audio_tokens)
    use_audio = include_audio and len(_audio_list) > 0

    def fmt_leader():
        return D["leader_motion_open"] + " " + " ".join(D["ih_token"].format(int(t)) for t in leader_tokens) + " " + D["leader_motion_close"]

    def fmt_follower():
        return D["follower_motion_open"] + " " + " ".join(D["ih_token"].format(int(t)) for t in follower_tokens) + " " + D["follower_motion_close"]

    def fmt_relationship():
        return D["relationship_open"] + " " + " ".join(D["rel_token"].format(int(t)) for t in relationship_tokens) + " " + D["relationship_close"]

    def fmt_audio():
        return D["audio_open"] + " " + " ".join(D["audio_token"].format(int(t)) for t in _audio_list) + " " + D["audio_close"]

    # Order: caption/texts → audio → input motion → input relation. Each block gets a textual label.
    def build_input_body(caption_txt=None, audio_block=None, motion_blocks=None, relation_block=None):
        parts = []
        if caption_txt:
            parts.append(caption_txt.strip())
        if audio_block:
            parts.append(D["label_music"] + audio_block)
        for label, content in motion_blocks or []:
            parts.append(label + content)
        if relation_block:
            label, content = relation_block
            parts.append(label + content)
        return "\n\n".join(parts) + "\n\n"

    if task == "leader_rel_to_follower":
        instruction = D.get("instruction_leader_rel_to_follower_audio", D["instruction_leader_rel_to_follower"]) if use_audio else D["instruction_leader_rel_to_follower"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_leader_motion"], fmt_leader())], relation_block=(D["label_relationship"], fmt_relationship()))
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_follower_motion"] + D["follower_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in follower_tokens) + " " + D["follower_motion_close"]

    elif task == "follower_rel_to_leader":
        instruction = D.get("instruction_follower_rel_to_leader_audio", D["instruction_follower_rel_to_leader"]) if use_audio else D["instruction_follower_rel_to_leader"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_follower_motion"], fmt_follower())], relation_block=(D["label_relationship"], fmt_relationship()))
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_leader_motion"] + D["leader_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in leader_tokens) + " " + D["leader_motion_close"]

    elif task == "caption_leader_rel_to_follower":
        instruction = D.get("instruction_caption_leader_rel_to_follower_audio", D["instruction_caption_leader_rel_to_follower"]) if use_audio else D["instruction_caption_leader_rel_to_follower"]
        if caption is None:
            caption_parts = []
            if level:
                caption_parts.append(f"Level: {level}.")
            move_summary = _format_move_summary_for_prompt(move_annotations or [])
            if move_summary:
                caption_parts.append(f"Moves: {move_summary}")
            caption = " ".join(caption_parts) if caption_parts else "Salsa pair motion."
        input_body = build_input_body(caption_txt=caption.strip(), audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_leader_motion"], fmt_leader())], relation_block=(D["label_relationship"], fmt_relationship()))
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_follower_motion"] + D["follower_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in follower_tokens) + " " + D["follower_motion_close"]

    elif task == "caption_follower_rel_to_leader":
        instruction = D.get("instruction_caption_follower_rel_to_leader_audio", D["instruction_caption_follower_rel_to_leader"]) if use_audio else D["instruction_caption_follower_rel_to_leader"]
        if caption is None:
            caption_parts = []
            if level:
                caption_parts.append(f"Level: {level}.")
            move_summary = _format_move_summary_for_prompt(move_annotations or [])
            if move_summary:
                caption_parts.append(f"Moves: {move_summary}")
            caption = " ".join(caption_parts) if caption_parts else "Salsa pair motion."
        input_body = build_input_body(caption_txt=caption.strip(), audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_follower_motion"], fmt_follower())], relation_block=(D["label_relationship"], fmt_relationship()))
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_leader_motion"] + D["leader_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in leader_tokens) + " " + D["leader_motion_close"]

    elif task == "pair_to_relationship":
        instruction = D.get("instruction_pair_to_relationship_audio", D["instruction_pair_to_relationship"]) if use_audio else D["instruction_pair_to_relationship"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_leader_motion"], fmt_leader()), (D["label_follower_motion"], fmt_follower())])
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_relationship"] + D["relationship_open"] + " "
        target_text = " ".join(D["rel_token"].format(int(t)) for t in relationship_tokens) + " " + D["relationship_close"]

    elif task == "caption_to_leader":
        instruction = D.get("instruction_caption_to_leader_audio", D["instruction_caption_to_leader"]) if use_audio else D["instruction_caption_to_leader"]
        if caption is None:
            caption_parts = []
            if level:
                caption_parts.append(f"Level: {level}.")
            move_summary = _format_move_summary_for_prompt(move_annotations or [])
            if move_summary:
                caption_parts.append(f"Moves: {move_summary}")
            caption = " ".join(caption_parts) if caption_parts else "Salsa pair motion."
        input_body = build_input_body(caption_txt=caption.strip(), audio_block=fmt_audio() if use_audio else None)
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_leader_motion"] + D["leader_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in leader_tokens) + " " + D["leader_motion_close"]

    elif task == "caption_to_follower":
        instruction = D.get("instruction_caption_to_follower_audio", D["instruction_caption_to_follower"]) if use_audio else D["instruction_caption_to_follower"]
        if caption is None:
            caption_parts = []
            if level:
                caption_parts.append(f"Level: {level}.")
            move_summary = _format_move_summary_for_prompt(move_annotations or [])
            if move_summary:
                caption_parts.append(f"Moves: {move_summary}")
            caption = " ".join(caption_parts) if caption_parts else "Salsa pair motion."
        input_body = build_input_body(caption_txt=caption.strip(), audio_block=fmt_audio() if use_audio else None)
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_follower_motion"] + D["follower_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in follower_tokens) + " " + D["follower_motion_close"]

    elif task == "leader_to_follower":
        instruction = D.get("instruction_leader_to_follower_audio", D["instruction_leader_to_follower"]) if use_audio else D["instruction_leader_to_follower"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_leader_motion"], fmt_leader())])
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_follower_motion"] + D["follower_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in follower_tokens) + " " + D["follower_motion_close"]

    elif task == "follower_to_leader":
        instruction = D.get("instruction_follower_to_leader_audio", D["instruction_follower_to_leader"]) if use_audio else D["instruction_follower_to_leader"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_follower_motion"], fmt_follower())])
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_leader_motion"] + D["leader_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in leader_tokens) + " " + D["leader_motion_close"]

    elif task == "motion_completion_leader":
        instruction = D.get("instruction_motion_completion_leader_audio", D["instruction_motion_completion_leader"]) if use_audio else D["instruction_motion_completion_leader"]
        n = max(1, int(len(leader_tokens) * completion_split_ratio))
        partial_tokens, rest_tokens = leader_tokens[:n], leader_tokens[n:]
        partial_str = D["leader_motion_open"] + " " + " ".join(D["ih_token"].format(int(t)) for t in partial_tokens) + " " + D["leader_motion_close"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_leader_motion"], partial_str)])
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_leader_motion"] + D["leader_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in rest_tokens) + " " + D["leader_motion_close"]

    elif task == "motion_completion_follower":
        instruction = D.get("instruction_motion_completion_follower_audio", D["instruction_motion_completion_follower"]) if use_audio else D["instruction_motion_completion_follower"]
        n = max(1, int(len(follower_tokens) * completion_split_ratio))
        partial_tokens, rest_tokens = follower_tokens[:n], follower_tokens[n:]
        partial_str = D["follower_motion_open"] + " " + " ".join(D["ih_token"].format(int(t)) for t in partial_tokens) + " " + D["follower_motion_close"]
        input_body = build_input_body(audio_block=fmt_audio() if use_audio else None, motion_blocks=[(D["label_follower_motion"], partial_str)])
        prompt_text = D["system_prompt"] + instruction + D["input_section"] + input_body + D["response_header"] + D["label_follower_motion"] + D["follower_motion_open"] + " "
        target_text = " ".join(D["ih_token"].format(int(t)) for t in rest_tokens) + " " + D["follower_motion_close"]

    else:
        raise ValueError(f"Unknown task: {task}. Use one of: {INTERHUMAN_TASKS}")

    return prompt_text, target_text

