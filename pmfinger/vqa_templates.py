"""Natural-language templates for the VQA generator. Several paraphrases per task and language;
one is drawn at random per sample. Add paraphrases freely: placeholders are filled with str.format."""

FINGER = {
    "zh": ["拇指", "食指", "中指", "无名指", "小指"],
    "en": ["thumb", "index finger", "middle finger", "ring finger", "little finger"],
}
HAND = {
    "zh": ["左手", "右手", "双手"],
    "en": ["the left hand", "the right hand", "both hands"],
}

T = {
    "finger_of_note": {
        "zh": [
            ("图中{hand}按下 {key} 键用的是哪根手指？", "{finger}（{fnum} 指）。"),
            ("这一帧里，{key} 是被{hand}的哪根手指按下的？", "{hand}的{finger}。"),
            ("请看图：{hand}正在按下 {key}，按键的是哪根手指？", "{finger}。"),
        ],
        "en": [
            ("In this frame, which finger of {hand} presses {key}?", "The {finger} ({fnum})."),
            ("{hand} is pressing {key} here. Which finger does it use?", "The {finger}."),
        ],
    },
    "finger_hand_of_note": {
        "zh": [
            ("图中 {key} 键是哪只手的哪根手指按下的？", "{hand}的{finger}。"),
            ("这一帧按下 {key} 的是哪只手、哪根手指？", "{hand}{finger}（{fnum} 指）。"),
        ],
        "en": [
            ("Which hand and which finger press {key} in this frame?", "The {finger} of {hand}."),
        ],
    },
    "key_of_finger": {
        "zh": [
            ("此刻{hand}的{finger}按在哪个键上？", "{key}。"),
            ("图中{hand}{finger}正在按哪个音？", "{key}。"),
        ],
        "en": [
            ("Which key is the {finger} of {hand} holding down right now?", "{key}."),
            ("In this frame, what note is {hand}'s {finger} pressing?", "{key}."),
        ],
    },
    "sequence_between": {
        "zh": [
            ("从第一张图到第二张图之间，{hand}依次弹了哪些音？分别用的是哪根手指？", "{seq}"),
            ("两帧之间{hand}按时间顺序按下了哪些键，各用哪根手指？", "{seq}"),
        ],
        "en": [
            ("Between the first and the second frame, which notes did {hand} play, in order, and with which fingers?", "{seq}"),
            ("List, in order, the keys {hand} pressed between the two frames and the finger used for each.", "{seq}"),
        ],
    },
    "count_between": {
        "zh": [
            ("第一张图和第二张图之间，{hand}一共按下了几个键？", "{count} 个。"),
            ("两帧之间{hand}按键多少次？", "{count} 次。"),
        ],
        "en": [
            ("How many keys did {hand} press between the first and the second frame?", "{count}."),
        ],
    },
    "hand_shift": {
        "zh": [
            ("从第一张图到第二张图，{hand}朝哪个方向移动了，大约移动了几个白键？", "{shift}"),
            ("对比两张图，{hand}的位置发生了怎样的变化？", "{shift}"),
        ],
        "en": [
            ("From the first frame to the second, in which direction and by how many white keys did {hand} move?", "{shift}"),
        ],
    },
    "pressed_at": {
        "zh": [
            ("图中此刻哪些手指按在琴键上？分别按的是哪个键？", "{state}"),
            ("这一帧双手各有哪些手指正按着键？", "{state}"),
        ],
        "en": [
            ("Which fingers are holding keys down in this frame, and which keys?", "{state}"),
        ],
    },
    "hand_register": {
        "zh": [
            ("图中{hand}大约覆盖了哪些键？", "{range}"),
            ("{hand}现在位于键盘的哪个区域？", "{range}"),
        ],
        "en": [
            ("Which range of keys does {hand} cover in this frame?", "{range}"),
        ],
    },
    "predict_fingering": {
        "zh": [
            ("图中为当前手的状态。接下来{hand}要依次弹 {keys}，最合理的指法是什么？", "{fingers}"),
            ("{hand}接下来要弹 {keys}。结合图中手的位置，每个音应该用哪根手指？", "{fingers}"),
        ],
        "en": [
            ("The image shows the current hand state. Next, {hand} will play {keys}. Which fingers should it use?", "{fingers}"),
        ],
    },
    "predict_hand_pos": {
        "zh": [
            ("图中为当前状态。{hand}接下来依次弹 {keys}，弹完后{hand}大约会在键盘的什么位置？", "{range}"),
        ],
        "en": [
            ("Given this frame, {hand} is about to play {keys}. Where on the keyboard will it be afterwards?", "{range}"),
        ],
    },
}

PHRASE = {
    "zh": dict(
        seq_empty="这段时间{hand}没有按下任何键。",
        shift_none="基本没有移动。",
        shift_right="向右（高音方向）移动了约 {n} 个白键。",
        shift_left="向左（低音方向）移动了约 {n} 个白键。",
        range="大约覆盖 {lo} 到 {hi}，中心在 {mid} 附近。",
        hand_none="没有手指按键",
        press_item="{finger}按 {key}",
        state="{hand}：{items}",
        state_sep="；",
        item_sep="、",
        arrow=" → ",
        options="选项：",
        mc_answer="{letter}. {text}",
    ),
    "en": dict(
        seq_empty="{hand} did not press any key in this interval.",
        shift_none="It stayed roughly in place.",
        shift_right="It moved right (toward the treble) by about {n} white keys.",
        shift_left="It moved left (toward the bass) by about {n} white keys.",
        range="Roughly from {lo} to {hi}, centred near {mid}.",
        hand_none="no finger down",
        press_item="{finger} on {key}",
        state="{hand}: {items}",
        state_sep="; ",
        item_sep=", ",
        arrow=" -> ",
        options="Options:",
        mc_answer="{letter}. {text}",
    ),
}
