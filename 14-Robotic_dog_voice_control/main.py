import os
import json
import subprocess
from vosk import Model, KaldiRecognizer
from difflib import SequenceMatcher
import serial
import time
import threading

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
model_path = "vosk-model-small-en-in-0.4"
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)
ser = serial.Serial("/dev/ttyUSB0", 115200)

MIC_DEVICE = "sysdefault:CARD=DLM3538CB"
MIC_RETRY_INTERVAL = 3  # seconds between retries when mic is missing

# ──────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────
active_group = None
wake_word_detected = False
sleep_requested = False
last_command_time = 0
COMMAND_COOLDOWN = 2.0

# ──────────────────────────────────────────────
# Serial command sender
# ──────────────────────────────────────────────

def send_cmd(var, val):
    dataCMD = json.dumps({'var': var, 'val': val})
    ser.write(dataCMD.encode())

MOVE_DURATION = 1.5

# ──────────────────────────────────────────────
# Robot movement — non-blocking (threaded)
# ──────────────────────────────────────────────

def _run_timed_move(start_val, label):
    send_cmd("move", start_val)
    print(f"Robot: {label}")
    time.sleep(MOVE_DURATION)
    send_cmd("move", 3)
    send_cmd("move", 6)

def forward():
    threading.Thread(target=_run_timed_move, args=(1, "forward"), daemon=True).start()

def backward():
    threading.Thread(target=_run_timed_move, args=(5, "backward"), daemon=True).start()

def stop_move():
    send_cmd("move", 3)
    send_cmd("move", 6)
    print("Robot: stop")

# ──────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────

def steady():      send_cmd("funcMode", 1); print("Robot: steady")
def handshake():   send_cmd("funcMode", 3); print("Robot: handshake")
def stop_action(): send_cmd("funcMode", 0); print("Robot: stop action")

# ──────────────────────────────────────────────
# Lights
# ──────────────────────────────────────────────

def light_off():     send_cmd("light", 0); print("Robot: lights off")
def light_blue():    send_cmd("light", 1); print("Robot: blue")
def light_red():     send_cmd("light", 3); print("Robot: red")
def light_green():   send_cmd("light", 2); print("Robot: green")
def light_yellow():  send_cmd("light", 4); print("Robot: yellow")
def light_cyan():    send_cmd("light", 5); print("Robot: cyan")
def light_magenta(): send_cmd("light", 6); print("Robot: magenta")
def light_cyber():   send_cmd("light", 7); print("Robot: cyber")

# ──────────────────────────────────────────────
# Buzzer
# ──────────────────────────────────────────────

def buzzer_on():  send_cmd("buzzer", 1); print("Robot: buzzer on")
def buzzer_off(): send_cmd("buzzer", 0); print("Robot: buzzer off")

def beep(count):
    for i in range(count):
        send_cmd("buzzer", 1)
        time.sleep(0.1)
        send_cmd("buzzer", 0)
        if i < count - 1:
            time.sleep(0.1)

# ──────────────────────────────────────────────
# Group switch config
# ──────────────────────────────────────────────

GROUP_SWITCH = {
    "movement": {
        "keywords": ["movement", "move", "walking"],
        "aliases":  ["moving", "walk", "motion", "locomotion"],
        "threshold": 0.60,
    },
    "light": {
        "keywords": ["light", "lights", "color", "colour"],
        "aliases":  ["led", "colors", "colours", "lighting"],
        "threshold": 0.60,
    },
    "sound": {
        "keywords": ["sound", "buzzer", "beep"],
        "aliases":  ["noise", "horn", "audio"],
        "threshold": 0.75,
    },
}

# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────

COMMANDS = {
    # Movement
    "forward": {
        "group": "movement",
        "keywords": ["forward", "go forward"],
        "aliases": ["for word", "foreword", "ford", "go ahead",
                    "move forward", "front", "forwards", "jack come here"],
        "action": forward, "threshold": 0.55,
    },
    "backward": {
        "group": "movement",
        "keywords": ["backward", "go back"],
        "aliases": ["back word", "backwards", "move back", "reverse",
                    "back", "go backward", "back up"],
        "action": backward, "threshold": 0.55,
    },
    "stop": {
        "group": "movement",
        "keywords": ["stop"],
        "aliases": ["top", "stuff", "stock", "stoop", "halt", "freeze"],
        "action": stop_move, "threshold": 0.55,
    },
    # Actions
    "steady": {
        "group": "movement",
        "keywords": ["steady"],
        "aliases": ["study", "steddy", "said he", "stead", "stay",
                    "stady", "stand", "standy", "balance"],
        "action": steady, "threshold": 0.55,
    },
    "handshake": {
        "group": "movement",
        "keywords": ["handshake"],
        "aliases": ["hand shake", "handshaking", "hand check",
                    "hang shake", "handsake", "shake hand", "shake",
                    "handshakes", "and shake"],
        "action": handshake, "threshold": 0.55,
    },
    "stop action": {
        "group": "movement",
        "keywords": ["stop action", "cancel"],
        "aliases": ["stop trick", "no action", "cancel action"],
        "action": stop_action, "threshold": 0.55,
    },
    # Lights
    "blue": {
        "group": "light",
        "keywords": ["blue", "blue light"],
        "aliases": ["blew", "bloo", "blue color"],
        "action": light_blue, "threshold": 0.60,
    },
    "red": {
        "group": "light",
        "keywords": ["red", "red light"],
        "aliases": ["read", "red color"],
        "action": light_red, "threshold": 0.60,
    },
    "green": {
        "group": "light",
        "keywords": ["green", "green light"],
        "aliases": ["grin", "green color"],
        "action": light_green, "threshold": 0.60,
    },
    "yellow": {
        "group": "light",
        "keywords": ["yellow", "yellow light"],
        "aliases": ["yell oh", "yellow color"],
        "action": light_yellow, "threshold": 0.60,
    },
    "dark": {
        "group": "light",
        "keywords": ["dark", "no light", "off"],
        "aliases": ["turn off", "lite off", "black"],
        "action": light_off, "threshold": 0.55,
    },
    # Buzzer
    "buzz": {
        "group": "sound",
        "keywords": ["beep", "buzz", "honk", "horn"],
        "aliases": ["make noise", "noise", "ring"],
        "action": buzzer_on, "threshold": 0.55,
    },
    "quiet": {
        "group": "sound",
        "keywords": ["silent", "stop beep", "stop buzz"],
        "aliases": ["no beep", "no noise", "mute", "shut up"],
        "action": buzzer_off, "threshold": 0.55,
    },
}

# ──────────────────────────────────────────────
# Wake / Sleep words
# ──────────────────────────────────────────────

WAKE_WORD = {
    "keywords": ["activate"],
    "aliases": ["active", "act of eight", "activates", "hey robot",
                "hello robot", "wake up", "hey jack"],
    "threshold": 0.50,
}

SLEEP_WORD = {
    "keywords": ["sleep", "go to sleep"],
    "aliases": ["asleep", "slip", "sleet", "sleeping",
                "good night", "goodbye"],
    "threshold": 0.55,
}

# ──────────────────────────────────────────────
# Fuzzy matching helpers
# ──────────────────────────────────────────────

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def word_matches(text, config):
    text_lower = text.lower().strip()
    words = text_lower.split()
    threshold = config["threshold"]

    for kw in config["keywords"] + config.get("aliases", []):
        if kw in text_lower:
            return True

    for word in words:
        for kw in config["keywords"]:
            if similarity(word, kw) >= threshold:
                return True

    for alias in config.get("aliases", []):
        if similarity(text_lower, alias) >= threshold:
            return True

    return False


def match_command(text):
    global active_group
    if active_group is None:
        return None

    best_cmd = None
    best_score = 0.0
    text_lower = text.lower().strip()
    words = text_lower.split()

    for cmd_name, config in COMMANDS.items():
        if config["group"] != active_group:
            continue

        score = 0.0

        for kw in config["keywords"] + config["aliases"]:
            if kw in text_lower:
                score = max(score, 0.9 + len(kw) / 100.0)

        for word in words:
            for kw in config["keywords"]:
                s = similarity(word, kw)
                if s >= config["threshold"]:
                    score = max(score, s)

        for alias in config["aliases"]:
            s = similarity(text_lower, alias)
            if s >= config["threshold"]:
                score = max(score, s)

        if score > best_score:
            best_score = score
            best_cmd = cmd_name

    return best_cmd


def check_group_switch(text, is_final):
    global active_group

    if active_group == "light":
        color_words = {"blue", "blew", "bloo", "red", "read", "green", "grin",
                       "yellow", "cyan", "scion", "magenta", "pink", "purple",
                       "cyber", "rainbow", "dark", "black", "off"}
        if any(w in color_words for w in text.lower().split()):
            return False

    if not is_final:
        return False

    for group_name, cfg in GROUP_SWITCH.items():
        if word_matches(text, cfg):
            if active_group == group_name:
                print(f"  Already in [{group_name.upper()}] group")
                return True

            prev = active_group
            active_group = group_name

            print("=" * 40)
            if prev:
                print(f"  SWITCHED: [{prev.upper()}] → [{group_name.upper()}]")
            else:
                print(f"  GROUP: [{group_name.upper()}] ACTIVE")
            cmds = [n for n, c in COMMANDS.items() if c["group"] == group_name]
            print(f"  Available: {', '.join(cmds)}")
            print("=" * 40)
            beep(1)
            return True

    return False


def print_active_group():
    if active_group:
        cmds = [name for name, cfg in COMMANDS.items()
                if cfg["group"] == active_group]
        print(f"  Available: {', '.join(cmds)}")
    else:
        print("  No group active — say 'movement', 'light', or 'sound'")

# ──────────────────────────────────────────────
# Audio pipeline — returns None if mic not found
# ──────────────────────────────────────────────

def get_audio_stream():
    # Check if the card is actually present before launching arecord
    check = subprocess.run(
        ["arecord", "-l"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    card_name = MIC_DEVICE.split("CARD=")[-1]  # "DLM3538CB"
    if card_name not in check.stdout.decode() + check.stderr.decode():
        return None  # mic not plugged in yet

    command = [
        "arecord", "-D", MIC_DEVICE,
        "-f", "S16_LE", "-r", "48000", "-c", "2", "-t", "raw"
    ]
    sox_command = [
        "sox", "-t", "raw", "-b", "16", "-e", "signed-integer",
        "-c", "2", "-r", "48000", "-",
        "-t", "raw", "-b", "16", "-e", "signed-integer",
        "-c", "1", "-r", "16000", "-"
    ]
    try:
        arecord_proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        sox_proc = subprocess.Popen(
            sox_command,
            stdin=arecord_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Brief pause then check arecord didn't immediately die
        time.sleep(0.5)
        if arecord_proc.poll() is not None:
            sox_proc.terminate()
            return None
        return sox_proc
    except Exception as e:
        print(f"  Stream error: {e}")
        return None

# ──────────────────────────────────────────────
# Audio processing
# ──────────────────────────────────────────────

def process_audio_data(in_data):
    global wake_word_detected, sleep_requested, active_group, last_command_time

    is_final = recognizer.AcceptWaveform(in_data)

    if is_final:
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
    else:
        partial = json.loads(recognizer.PartialResult())
        text = partial.get("partial", "")

    if not text.strip():
        return

    if not is_final:
        if wake_word_detected:
            check_group_switch(text, is_final)
        return

    # ── Not awake ──
    if not wake_word_detected:
        if word_matches(text, WAKE_WORD):
            wake_word_detected = True
            sleep_requested = False
            active_group = None
            print("=" * 40)
            print("  AWAKE!")
            print("  Say 'movement', 'light', or 'sound'")
            print("  to activate a group.")
            print("=" * 40)
            beep(1)
        return

    # ── Sleep check ──
    if word_matches(text, SLEEP_WORD):
        sleep_requested = True
        wake_word_detected = False
        active_group = None
        print("=" * 40)
        print("  SLEEPING... All disabled.")
        print("  Say 'activate' to wake.")
        print("=" * 40)
        beep(3)
        return

    # ── Group switch ──
    if check_group_switch(text, is_final):
        return

    if active_group is None:
        print("    No group active — say 'movement', 'light', or 'sound'")
        return

    if not is_final:
        return

    # ── Cooldown ──
    now = time.time()
    if now - last_command_time < COMMAND_COOLDOWN:
        print(f"    (cooldown — ignoring '{text}')")
        return

    print(f"Heard: '{text}'")
    cmd = match_command(text)
    if cmd:
        last_command_time = now
        print(f">>> [{active_group}] {cmd}")
        COMMANDS[cmd]["action"]()
    else:
        print(f"    (no match in [{active_group}] group)")

# ──────────────────────────────────────────────
# Main — hot-plug aware loop
# ──────────────────────────────────────────────

def start_listening():
    global sleep_requested

    print("=" * 50)
    print("  WAVESHARE ROBOT DOG — VOICE CONTROL")
    print("=" * 50)
    print()
    print("  1. Say 'activate' to wake up")
    print("  2. Say a group name to switch to it:")
    print()
    print("  'movement' — walk + tricks")
    print("     forward, backward, stop")
    print("     steady, handshake, stop action")
    print()
    print("  'light' — LED colors")
    print("     blue, red, green, yellow, dark")
    print()
    print("  'sound' — buzzer control")
    print("     beep/buzz/honk, silent/mute")
    print()
    print("  Switch anytime by saying the group name.")
    print("  Say 'sleep' to deactivate everything.")
    print()
    print("  Buzzer feedback:")
    print("     1 beep = switched / wake")
    print("     3 beeps = sleep")
    print("=" * 50)

    while True:  # outer loop — reconnects if mic is lost/unplugged
        stream = None

        # ── Wait until mic is available ──
        while stream is None:
            print(f"  Waiting for mic ({MIC_DEVICE})...")
            stream = get_audio_stream()
            if stream is None:
                time.sleep(MIC_RETRY_INTERVAL)

        print("  Mic connected — listening!")

        try:
            while True:  # inner loop — normal operation
                data = stream.stdout.read(4000)

                # Empty read means the stream died (mic unplugged)
                if not data:
                    print("\n  Mic disconnected — waiting to reconnect...")
                    break

                process_audio_data(data)

                if sleep_requested:
                    sleep_requested = False

        except KeyboardInterrupt:
            print("\nShutting down...")
            if stream:
                stream.terminate()
            return  # clean exit

        except Exception as e:
            print(f"  Stream error: {e}")

        finally:
            # Clean up the dead stream before retrying
            try:
                if stream:
                    stream.terminate()
            except Exception:
                pass

        # Small delay before trying to reconnect
        time.sleep(MIC_RETRY_INTERVAL)


if __name__ == "__main__":
    start_listening()
