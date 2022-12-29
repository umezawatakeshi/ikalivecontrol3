# IkaLiveControl3
# Copyright (c) 2022 UMEZAWA Takeshi
# Licensed under GNU GPL version 2 or later

# キャプボあるいは動画ファイルを開いて観戦動画配信支援

import cv2
import numpy as np
import argparse
from time import sleep
import yaml
import pyautogui
import os

OBS_SCENE_LOBBY_HOTKEY = ["scrolllock"]
OBS_SCENE_BATTLE_HOTKEY = ["pause"]
OBS_SCENE_CHANGE_DELAY = 1

RULES = ["nawabari", "area", "yagura", "hoko", "asari"]
# ステージリストは さんぽ や プラベ のステージ選択画面と同じ並びにしてある
STAGES = [
	"yunohana", "gonzui", "yagara", "mategai",
	"namerou", "masaba", "kinmedai", "mahimahi",
	"ama", "chouzame", "zatou", "sumeshi",
	"kusaya", "hirame"]

PHASE_WAIT_JOIN = 0  # 「メンバーの合流を待っています」を検出した後の状態
                     #   - 「親がチームを決めています」になった後も PHASE_WAIT_JOIN のままだが気にしてはいけない
                     #   - 「メンバーの準備を待っています」を検出したら PHASE_WAIT_READY へ
PHASE_WAIT_READY = 1 # 「メンバーの準備を待っています」を検出した後の状態
                     #   - ルールとステージの検出処理が走る
                     #   - 試合のオープニングを検出したら PHASE_OPENING へ
PHASE_OPENING = 2    # 試合のオープニング（ルールとステージが表示されるやつ）を検出した後の状態
                     #   - ルールとステージの検出がまだだったら検出処理が走る
                     #   - 試合の開始を検出したら PHASE_GAME へ
                     #     正確には試合が開始して1秒経過したことを検出したら。
                     #     テンプレートマッチングとしては残り時間の分の部分が 4 または 2 になっていることを検出する
PHASE_GAME = 3       # 試合の開始を検出した後の状態
                     #   - Finish! を検出したら PHASE_FINISH へ
PHASE_FINISHED = 4   # Finish! を検出した後の状態
                     #   - 勝利/敗北 を検出したら PHASE_RESULT へ
PHASE_RESULT = 5     # 勝利/敗北 を検出した後の状態

                     # どの phase でも、「メンバーの合流を待っています」を検出したら PHASE_WAIT_JOIN へ
                     #   - 試合結果を検出する前に「メンバーの合流を待っています」を検出したら無効試合扱い

SCENE_LOBBY = 0
SCENE_BATTLE = 1

current_phase = PHASE_WAIT_JOIN
current_scene = SCENE_LOBBY

# 画像のテンプレートマッチングを行うクラス
# 640x360 にリサイズして指定した閾値で白黒二値化した状態で、指定した範囲についてマッチングを行う
class Matcher:
	BLACK = 4
	DARK = 32
	LIGHT = 224
	WHITE = 252
	THRESHOLDS = [BLACK, DARK, LIGHT, WHITE]

	@staticmethod
	def pre_filter(src, threshold):
		tmp = src
		# BLACK と WHITE は完全な黒および白を検出する
		#if threshold == Matcher.BLACK or threshold == Matcher.WHITE:
		#	_, tmp = cv2.threshold(tmp, threshold, 255, cv2.THRESH_BINARY)
		tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
		_, tmp = cv2.threshold(tmp, threshold, 255, cv2.THRESH_BINARY)
		return tmp

	def __init__(self, filename, top, left, bottom, right, threshold):
		self.top = top
		self.left = left
		self.bottom = bottom
		self.right = right
		self.pixels = (bottom - top) * (right - left)
		self.threshold = threshold

		tmp = cv2.imread(filename)
		tmp = Matcher.pre_filter(tmp, threshold)
		tmp = cv2.resize(tmp, (640, 360), cv2.INTER_CUBIC)
		tmp = self.crop(tmp)
		self.img = tmp

	def crop(self, src):
		return src[self.top:self.bottom, self.left:self.right]

	# マッチしたかどうかを返す
	def match(self, src, return_diff=False):
		tgt = self.crop(src[self.threshold])
		diff = cv2.absdiff(tgt, self.img).sum() * 1.0 / self.pixels
		matched = diff < args.matcher_threshold
		if return_diff:
			return (matched, diff)
		else:
			return matched

	# Matcher のリストを与えて、どれかとマッチしたかどうかを返す
	@staticmethod
	def list_match(matchers, src):
		for matcher in matchers:
			if  matcher.match(src):
				return True
		return False

def cv2putText(img, text, color, org):
	cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, 1, color, 5)
	cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

lobby_wait_join_matcher = Matcher("templates/lobby_wait_join.png", 13, 240, 33, 400, Matcher.LIGHT)
lobby_wait_ready_matchers = []
lobby_wait_ready_matchers.append(Matcher("templates/lobby_wait_ready_0.png", 13, 240, 33, 400, Matcher.LIGHT))
lobby_wait_ready_matchers.append(Matcher("templates/lobby_wait_ready_1.png", 24, 348, 37, 406, Matcher.LIGHT))
result_win_matcher = Matcher("templates/result_win.png", 17, 10, 45, 100, Matcher.LIGHT)
result_lose_matcher = Matcher("templates/result_lose.png", 17, 10, 45, 100, Matcher.LIGHT)
result_nogame_matcher = Matcher("templates/result_nogame.png", 81, 402, 93, 475, Matcher.LIGHT)
result_matchers = {}
result_matchers["alpha"] = result_win_matcher
result_matchers["bravo"] = result_lose_matcher
result_matchers["nogame"] = result_nogame_matcher
started_matchers = []
started_matchers.append(Matcher("templates/started_0a.png", 18, 304, 31, 318, Matcher.LIGHT))
started_matchers.append(Matcher("templates/started_0b.png", 18, 304, 31, 318, Matcher.LIGHT))
started_matchers.append(Matcher("templates/started_1a.png", 18, 304, 31, 318, Matcher.LIGHT))
started_matchers.append(Matcher("templates/started_1b.png", 18, 304, 31, 318, Matcher.LIGHT))

opening_matchers = []

rule_matchers = {}
for r in RULES:
	rule_matchers[r] = []
	rule_matchers[r].append(Matcher("templates/lobby_rule_{0}.png".format(r), 137, 78, 153, 161, Matcher.DARK))
	matcher = Matcher("templates/opening_rule_{0}.png".format(r), 223, 243, 242, 396, Matcher.LIGHT)
	rule_matchers[r].append(matcher)
	opening_matchers.append(matcher)

stage_matchers = {}
for s in STAGES:
	stage_matchers[s] = []
	stage_matchers[s].append(Matcher("templates/lobby_stage_{0}.png".format(s), 195, 140, 210, 210, Matcher.LIGHT))
	# スメーシーは水面のきらめきが邪魔で Matcher.WHITE でも検出できない（いい感じに閾値を下回らない）ので、 Matcher.BLACK を併用する。
	# 同じく水面がきらめいているマヒマヒはセーフだったのだが。
	for t in [Matcher.WHITE, Matcher.BLACK]:
		stage_matchers[s].append(Matcher("templates/opening_stage_{0}.png".format(s), 327, 505, 347, 627, t))

blank_matcher = Matcher("templates/blank.png", 90, 160, 270, 480, Matcher.BLACK)
finish_matchers = []
finish_matchers.append(Matcher("templates/finish_0a.png", 215, 339, 249, 410, Matcher.DARK))
finish_matchers.append(Matcher("templates/finish_0b.png", 215, 339, 249, 410, Matcher.DARK))

rule_images = {}
for r in RULES:
	rule_images[r] = cv2.imread("images/rule_{0}.png".format(r), cv2.IMREAD_UNCHANGED)

stage_images = {}
for s in STAGES:
	stage_images[s] = cv2.imread("images/stage_{0}.png".format(s), cv2.IMREAD_UNCHANGED)
stage_unknown_image = cv2.imread("images/stage_unknown.png", cv2.IMREAD_UNCHANGED)

number_images = {}
for i in range(10):
	number_images["{}".format(i)] = cv2.imread("images/number_{0}.png".format(i), cv2.IMREAD_UNCHANGED)

ika_images = {}
for i in ["alpha", "bravo"]:
	ika_images[i] = cv2.imread("images/ika_{0}.png".format(i), cv2.IMREAD_UNCHANGED)

banner_lobby_image = cv2.imread("images/banner_lobby.png", cv2.IMREAD_UNCHANGED)
banner_battle_image = cv2.imread("images/banner_battle.png", cv2.IMREAD_UNCHANGED)

argparser = argparse.ArgumentParser()
argparser.add_argument('--video-file', dest='video_file_name', help='specify input video file')
argparser.add_argument('--video-capture', dest='video_capture_id', type=int, help='specify input video capture device id')
argparser.add_argument('--capture-size', dest='capture_size', type=str, help='video capture size in width,height')
argparser.add_argument('--capture-fps', dest='capture_fps', type=float, help='video capture fps')
argparser.add_argument('--progress-file', dest='progress_file_name', help='specify progress file')
argparser.add_argument('--control-obs', dest='control_obs', action='store_true', help='control OBS')
argparser.add_argument('--output-dir', dest='output_dir_name', help='specify image output directory')
argparser.add_argument('--test-matchers', dest='test_matchers', action='store_true', help='test matchers')
argparser.add_argument('--decimate-frames', dest='decimate_frames', action='store_true', help='decimate frames to decrease cpu load')
argparser.add_argument('--matcher-threshold', dest='matcher_threshold', type=float, default=2.0, help='template matching threshold')
args = argparser.parse_args()
#print(args)

if not args.video_file_name is None:
	cap = cv2.VideoCapture(args.video_file_name)
elif not args.video_capture_id is None:
	cap = cv2.VideoCapture(args.video_capture_id)
else:
	cap = cv2.VideoCapture(0)

if not args.capture_size is None:
	size = args.capture_size.split(",")
	if len(size) != 2:
		raise ValueError("'{}' is not size".format(args.capture_size))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(size[0]))
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(size[1]))
if not args.capture_fps is None:
	cap.set(cv2.CAP_PROP_FPS, args.capture_fps)

print("capture size = {}x{}".format(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1
print("frame rate = {:5.2f}fps".format(fps))
if args.decimate_frames and fps > 15.0:
	frame_interval =  int(fps / 12.5) # 29.97fps 系や 25fps 系だった場合に備えて 15 ではなく 12.5 で割る
	print("check every {} frames".format(frame_interval))

if args.progress_file_name is None:
	progress = {}
else:
	with open(args.progress_file_name) as file:
		progress = yaml.safe_load(file)
if not "teams" in progress or not isinstance(progress["teams"], dict):
	progress["teams"] = {}
teams = progress["teams"]
if not "alpha" in teams or not isinstance(teams["alpha"], str):
	teams["alpha"] = ""
if not "bravo" in teams or not isinstance(teams["bravo"], str):
	teams["bravo"] = ""
if not "alpha_short_image" in teams or not isinstance(teams["alpha_short_image"], str):
	teams["alpha_short_image"] = "images/team_alpha_short.png"
if not "bravo_short_image" in teams or not isinstance(teams["bravo_short_image"], str):
	teams["bravo_short_image"] = "images/team_bravo_short.png"
if not "alpha_long_image" in teams or not isinstance(teams["alpha_long_image"], str):
	teams["alpha_long_image"] = "images/team_alpha_long.png"
if not "bravo_long_image" in teams or not isinstance(teams["bravo_long_image"], str):
	teams["bravo_long_image"] = "images/team_bravo_long.png"
if not "games" in progress or not isinstance(progress["games"], list):
	progress["games"] = []
games = progress["games"]

alpha_long_image = cv2.imread(teams["alpha_long_image"], cv2.IMREAD_UNCHANGED)
bravo_long_image = cv2.imread(teams["bravo_long_image"], cv2.IMREAD_UNCHANGED)
alpha_short_image = cv2.imread(teams["alpha_short_image"], cv2.IMREAD_UNCHANGED)
bravo_short_image = cv2.imread(teams["bravo_short_image"], cv2.IMREAD_UNCHANGED)

# 画像を貼り付ける
# 貼り付ける画像が RGBA 画像である場合は、アルファブレンディングを行う。
# 制限事項: はみ出すような貼り付けを行おうとするとコケる。
def paste_image(oimg, iimg, y, x):
	h = iimg.shape[0]
	w = iimg.shape[1]
	if iimg.shape[2] == 3:
		oimg[y:y+h, x:x+w] = iimg
	elif oimg.shape[2] == 3:
		oimg[y:y+h, x:x+w] = oimg[y:y+h, x:x+w] * (1 - iimg[:, :, 3:] / 255) + iimg[:, :, :3] * (iimg[:, :, 3:] / 255)
	else:
		ialpha = iimg[:, :, 3:].astype(float)
		itransparency_scaled = 1 - ialpha / 255
		oalpha = oimg[y:y+h, x:x+w, 3:].astype(float)
		new_oalpha = iimg[:, :, 3:] + oalpha * itransparency_scaled
		new_orgb = (iimg[:, :, :3] * ialpha + oimg[y:y+h, x:x+w, :3] * itransparency_scaled * new_oalpha)
		new_orgb = np.divide(new_orgb, new_oalpha, out=np.zeros_like(new_orgb), where=new_oalpha!=0)
		oimg[y:y+h, x:x+w] = np.block([new_orgb, new_oalpha])

# 画像をグレースケール化する
# RGB 画像を与えた場合、それをグレースケール化した RGB 画像を返す。3つのチャンネルは同じ値になる。
# RGBA 画像を与えた場合、RGB 部分をグレースケール化した RGB 画像に元のアルファを結合した画像を返す。
def cvt_grayscale(img):
	if img.shape[2] == 3:
		return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
	else:
		return np.block([cv2.cvtColor(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), img[:, :, 3:]])

def draw_progress():
	lobby_frame[:] = 0
	battle_banner[:] = 0
	paste_image(lobby_frame, banner_lobby_image, 0, 0)
	paste_image(battle_banner, banner_battle_image, 0, 0)

	wins = {"alpha": 0, "bravo": 0}
	for game in games:
		if "result" in game and game["result"] != "nogame":
			wins[game["result"]] += 1
	s = "{}".format(wins["alpha"])
	for k in range(len(s)):
		paste_image(lobby_frame, number_images[s[k]], 72-48//2, 720+48-len(s)*16+32*k)
		paste_image(battle_banner, number_images[s[k]], 72-48//2, 240+48-len(s)*16+32*k)
	s = "{}".format(wins["bravo"])
	for k in range(len(s)):
		paste_image(lobby_frame, number_images[s[k]], 72-48//2, 1920-(720+48)-len(s)*16+32*k)
		paste_image(battle_banner, number_images[s[k]], 72-48//2, 1920-(240+48)-len(s)*16+32*k)
	paste_image(lobby_frame, alpha_long_image, 24, 0)
	paste_image(lobby_frame, bravo_long_image, 24, 1920-720)
	paste_image(battle_banner, alpha_short_image, 24, 0)
	paste_image(battle_banner, bravo_short_image, 24, 1920-240)

	numlist = 8
	base = max(0, len(games) - numlist)
	for i in range(numlist):
		j = i + base
		if j >= len(games):
			break
		top = 120+i*120
		game = games[j]
		nogame = "result" in game and game["result"] == "nogame"
		if "stage" in game:
			tmp = stage_images[game["stage"]]
			if nogame:
				tmp = cvt_grayscale(tmp)
		else:
			tmp = stage_unknown_image
		paste_image(lobby_frame, tmp, top, 8)
		if "rule" in game:
			tmp = rule_images[game["rule"]]
			if nogame:
				tmp = cvt_grayscale(tmp)
			paste_image(lobby_frame, tmp, top+8, 160)
		if "result" in game and game["result"] != "nogame":
			paste_image(lobby_frame, ika_images[game["result"]], top+60, 160)
		s = "{}".format(j+1)
		for k in range(len(s)):
			paste_image(lobby_frame, number_images[s[k]], top+8, 16+32*k)

	if not args.output_dir_name is None:
		cv2.imwrite(os.path.join(args.output_dir_name, "lobby_frame.tmp.png"), lobby_frame)
		cv2.imwrite(os.path.join(args.output_dir_name, "battle_banner.tmp.png"), battle_banner)

		# PermissionError: [WinError 5] アクセスが拒否されました。 が出ることがある（たぶん OBS が読み込みのみ共有で開いてる）のでリトライする。
		def replace_with_retry(src_base, dst_base):
			num_try = 10
			for i in range(num_try):
				try:
					os.replace(os.path.join(args.output_dir_name, src_base), os.path.join(args.output_dir_name, dst_base))
					break
				except PermissionError as e:
					#print(e)
					if i == num_try - 1:
						throw
					sleep(0.02)
		replace_with_retry("lobby_frame.tmp.png", "lobby_frame.png")
		replace_with_retry("battle_banner.tmp.png", "battle_banner.png")

	outframe[:] = 128
	paste_image(outframe, lobby_frame, 0, 0)
	paste_image(outframe, battle_banner, 1080, 0)

def write_progress():
	if not args.progress_file_name is None:
		with open(args.progress_file_name, "w") as file:
			yaml.dump(progress, file)
	draw_progress()
	#print(yaml.dump(progress))

lobby_frame = np.zeros((1080, 1920, 4), np.uint8) # RGBA, 1920x1080, 透明
battle_banner = np.zeros((120, 1920, 4), np.uint8) # RGBA, 1920x120, 透明
outframe = np.zeros((1200, 1920, 3), np.uint8) # RGB24, 1920x1200, 真っ黒
img = {}
draw_progress()

while True:
	for i in range(frame_interval):
		ret, captured = cap.read()
		if not ret:
			print("cannot read from capture")
			break
	if not ret:
		break
	#for i in Matcher.THRESHOLDS:
	#	img[i]  = cv2.resize(Matcher.pre_filter(captured, i), (640, 360), cv2.INTER_CUBIC)
	captured = cv2.resize(captured, (640, 360), cv2.INTER_CUBIC)
	for i in Matcher.THRESHOLDS:
		img[i]  = Matcher.pre_filter(captured, i)


	if current_phase == PHASE_WAIT_JOIN:
		if Matcher.list_match(lobby_wait_ready_matchers, img):
			print("detected wait_ready")
			current_phase = PHASE_WAIT_READY
			current_game = {}
			games.append(current_game)
			write_progress()

	elif current_phase == PHASE_WAIT_READY or current_phase == PHASE_OPENING:
		if current_phase == PHASE_WAIT_READY:
			if Matcher.list_match(opening_matchers, img):
				print("detected opening")
				current_phase = PHASE_OPENING
		else: # PHASE_OPENING
			if Matcher.list_match(started_matchers, img):
				print("detected started")
				current_phase = PHASE_GAME

		# stage_matchers と rule_matchers はロビー用とオープニング用で分けた方が処理負荷が下がるが、めんどくさいのでそうしていない。
		if not "stage" in current_game:
			for s in STAGES:
				if Matcher.list_match(stage_matchers[s], img):
					print("detected stage {}".format(s))
					current_game["stage"] = s
					write_progress()
					break
		if not "rule" in current_game:
			for r in RULES:
				if Matcher.list_match(rule_matchers[r], img):
					print("detected rule {}".format(r))
					current_game["rule"] = r
					write_progress()
					break

	elif current_phase == PHASE_GAME:
		if Matcher.list_match(finish_matchers, img):
			print("detected finish")
			current_phase = PHASE_FINISHED

	elif current_phase == PHASE_FINISHED:
		if not "result" in current_game:
			for i in list(result_matchers):
				if result_matchers[i].match(img):
					print("detected result {}".format(i))
					current_phase = PHASE_RESULT
					current_game["result"] = i
					write_progress()
					break

	else:
		pass # XXX

	if current_phase == PHASE_WAIT_READY:
		if current_scene != SCENE_BATTLE and blank_matcher.match(img):
			print("detected blank to battle")
			current_scene = SCENE_BATTLE
			if args.control_obs:
				sleep(OBS_SCENE_CHANGE_DELAY)
				pyautogui.hotkey(*OBS_SCENE_BATTLE_HOTKEY, interval=0.01)
	elif current_phase == PHASE_GAME or current_phase == PHASE_RESULT:
		if current_scene != SCENE_LOBBY and blank_matcher.match(img):
			print("detected blank to lobby")
			current_scene = SCENE_LOBBY
			if args.control_obs:
				sleep(OBS_SCENE_CHANGE_DELAY)
				pyautogui.hotkey(*OBS_SCENE_LOBBY_HOTKEY, interval=0.01)

	# 念のため、どの状態からも PHASE_WAIT_JOIN に戻れるようにしておく
	if current_phase != PHASE_WAIT_JOIN:
		if lobby_wait_join_matcher.match(img):
			print("detected wait_join")
			current_phase = PHASE_WAIT_JOIN
			if current_scene != SCENE_LOBBY:
				current_scene = SCENE_LOBBY
				if args.control_obs:
					pyautogui.hotkey(*OBS_SCENE_LOBBY_HOTKEY, interval=0.01)
			# ロビーに戻ってきたのに結果が検出されていない場合は無効試合扱いにしておく
			if not "result" in current_game:
				current_game["result"] = "nogame"
				write_progress()


	# 出力バッファに貼り付け
	outframe[180:540, 320:960]  = captured
	for i in range(3):
		outframe[540:900, 320:960,  i] = img[Matcher.DARK]
		outframe[540:900, 960:1600, i] = img[Matcher.BLACK]
		outframe[180:540, 960:1600, i] = img[Matcher.WHITE]

	# Matcher の動作確認
	if args.test_matchers:
		def match_color(b):
			if b:
				return (255, 0, 0)
			else:
				return (0, 0, 0)

		for i in range(len(STAGES)):
			s = ""
			mm = False
			for matcher in stage_matchers[STAGES[i]]:
				m, d = matcher.match(img, return_diff=True)
				mm = mm or m
				s = s + "{:6.2f} ".format(d)
			cv2putText(captured, s + STAGES[i], match_color(mm), (10, 15+i*15))
		for i in range(len(RULES)):
			s = ""
			mm = False
			for matcher in rule_matchers[RULES[i]]:
				m, d = matcher.match(img, return_diff=True)
				mm = mm or m
				s = s + "{:6.2f} ".format(d)
			cv2putText(captured, s + RULES[i], match_color(mm), (320, 15+i*15))
		m, d = lobby_wait_join_matcher.match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} wait_join".format(d), match_color(m), (320, 120))
		for i in range(len(lobby_wait_ready_matchers)):
			m, d = lobby_wait_ready_matchers[i].match(img, return_diff=True)
			cv2putText(captured, "{:6.2f} wait_ready_{}".format(d, i), match_color(m), (320, 135+i*15))
		m, d = result_win_matcher.match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} result_win".format(d), match_color(m), (320, 180))
		m, d = result_lose_matcher.match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} result_lose".format(d), match_color(m), (320, 195))
		m, d = result_nogame_matcher.match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} result_nogame".format(d), match_color(m), (320, 210))
		ma, da = started_matchers[0].match(img, return_diff=True)
		mb, db = started_matchers[1].match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} {:6.2f} started_{}".format(da, db, 0), match_color(ma or mb), (320, 240))
		ma, da = started_matchers[2].match(img, return_diff=True)
		mb, db = started_matchers[3].match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} {:6.2f} started_{}".format(da, db, 1), match_color(ma or mb), (320, 255))
		ma, da = finish_matchers[0].match(img, return_diff=True)
		mb, db = finish_matchers[1].match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} {:6.2f} finish_0".format(da, db, 0), match_color(ma or mb), (320, 270))
		m, d = blank_matcher.match(img, return_diff=True)
		cv2putText(captured, "{:6.2f} blank".format(d), match_color(m), (320, 300))
		outframe[180:540, 320:960]  = captured

	# ウィンドウに描画
	cv2.imshow('IkaLiveControl3', outframe)

	# q または ESC を押したら終了
	key = cv2.waitKey(1)
	if key & 0xff == ord('q') or key & 0xff == 0x1b:
		break
	# r を押したら画像を再出力（OBS がたまに置き換えられたファイルの読み込みに失敗するので）
	if key & 0xff == ord('r'):
		draw_progress()

cap.release()
cv2.destroyAllWindows()
