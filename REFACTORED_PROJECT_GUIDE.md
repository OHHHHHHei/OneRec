# MiniOneRec 閲嶆瀯鐗堣鏄庢枃妗?
## 1. 鏂囨。鐩殑

杩欎唤鏂囨。璇存槑褰撳墠閲嶆瀯鍚庣殑 MiniOneRec 椤圭洰濡備綍缁勭粐銆佸浣曡繍琛岋紝浠ュ強鏁存潯涓荤嚎娴佺▼濡備綍琛旀帴銆?
褰撳墠鐗堟湰鐨勭洰鏍囨槸锛?
- 淇濈暀鍘熼」鐩殑涓绘祦绋嬮€昏緫
- 灏嗕富閾捐矾杩佺Щ鍒?`src/minionerec`
- 淇濈暀鏃у叆鍙ｇ殑鍏煎灞?- 鐢ㄧ粺涓€ CLI + YAML 閰嶇疆椹卞姩鍚勯樁娈?
涓绘祦绋嬩繚鎸佷负锛?
```text
preprocess -> embed -> SID -> convert -> SFT -> RL -> evaluate
```

---

## 2. 褰撳墠椤圭洰缁撴瀯

### 2.1 鏍圭洰褰?
鏍圭洰褰曠幇鍦ㄤ富瑕佹壙鎷?4 绫昏亴璐ｏ細

- 鍏煎鏃у叆鍙?- 瀛樻斁閰嶇疆鏂囦欢
- 瀛樻斁鏁版嵁涓庝骇鐗?- 瀛樻斁鏂版簮鐮佸寘

鍏抽敭鐩綍濡備笅锛?
```text
MiniOneRec/
  archive/                 # 闈炰富绾夸唬鐮佸拰鏃ф枃妗ｅ綊妗?  assets/                  # 鍥剧墖璧勬簮
  config/                  # 鏃ч厤缃洰褰曪紝褰撳墠浠嶄繚鐣?  configs/                 # 鏂?YAML 閰嶇疆鐩綍
  data/                    # 鏁版嵁鐩綍
  minionerec/              # 椤跺眰鍏煎鍖咃紝鐢ㄤ簬鏈湴鐩存帴 import
  results/                 # 璇勪及缁撴灉
  rq/                      # 鏃?SID 瀛愮郴缁熺洰褰曪紝浠嶄繚鐣欏吋瀹瑰叆鍙?  scripts/                 # 鏂扮殑钖勫寘瑁呰剼鏈?  src/minionerec/          # 鏂颁富绾挎簮鐮?  tests/                   # 鏈湴濂戠害娴嬭瘯涓?smoke tests
  sft.py / rl.py / ...     # 鏃у叆鍙ｅ吋瀹瑰寘瑁?```

### 2.2 鏂颁富绾挎簮鐮?
涓荤嚎浠ｇ爜鐜板湪浣嶄簬锛?
- `src/minionerec`

鎸夎亴璐ｅ垝鍒嗗涓嬶細

- `cli/`
  - 缁熶竴鍛戒护鍏ュ彛
  - 鍚勯樁娈靛懡浠ゅ垎鍙?- `common/`
  - IO銆佽矾寰勩€佹棩蹇椼€乻eed銆乼okenizer 鍖呰
- `config/`
  - dataclass schema
  - YAML 鍔犺浇
  - CLI override 鍚堝苟
- `preprocess/`
  - Amazon18 / Amazon23 棰勫鐞?- `sid/`
  - text2emb
  - RQ 閲忓寲璁粌
  - SID 鐢熸垚
- `data/`
  - 鏁版嵁濂戠害
  - 鏁版嵁缂撳瓨
  - SFT / RL / Eval 鏁版嵁闆?  - convert
- `training/sft/`
  - tokenizer 鎵╁睍
  - SFT pipeline
- `training/rl/`
  - reward
  - RL pipeline
  - ReReTrainer
- `training/cf/`
  - SASRec 鐩稿叧
- `evaluation/`
  - 绾︽潫瑙ｇ爜
  - split / merge / metric
- `compat/`
  - 鏃у弬鏁板埌鏂伴厤缃殑鏄犲皠
  - 鏃у叆鍙ｅ吋瀹归€昏緫

### 2.3 褰掓。浠ｇ爜

浠ヤ笅鍐呭宸茬粡浠庝富閾捐矾鍓ョ锛?
- `archive/gpr`
- `archive/old_docs`

杩欎簺浠ｇ爜鍜屾枃妗ｄ笉浣滀负褰撳墠涓绘祦绋嬬殑涓€閮ㄥ垎锛屼絾淇濈暀浠ヤ究鍙傝€冦€?
---

## 3. 鐜板湪鐨勮繍琛屽叆鍙?
## 3.1 鏂扮粺涓€鍏ュ彛

鐜板湪鎺ㄨ崘浣跨敤缁熶竴 CLI锛?
```bash
python -m minionerec.cli.main <stage> --config <yaml> [overrides...]
```

鏀寔鐨?stage锛?
- `preprocess`
- `embed`
- `sid-train`
- `sid-generate`
- `convert`
- `sft`
- `rl`
- `evaluate`

绀轰緥锛?
```bash
python -m minionerec.cli.main sft --config configs/stages/sft/default.yaml
python -m minionerec.cli.main rl --config configs/stages/rl/default.yaml
python -m minionerec.cli.main evaluate --config configs/stages/evaluate/default.yaml
```

## 3.2 鍏煎鏃у叆鍙?
浠ヤ笅鏃у叆鍙ｄ粛鍙娇鐢細

- `sft.py`
- `rl.py`
- `evaluate.py`
- `convert_dataset.py`
- `split.py`
- `merge.py`
- `calc.py`

杩欎簺鏂囦欢鐜板湪鏈川涓婃槸鍏煎钖勫寘瑁咃紝浼氳浆璋冨埌鏂颁富绾垮疄鐜般€?
## 3.3 鏂拌剼鏈叆鍙?
鏂拌杽鍖呰鑴氭湰浣嶄簬锛?
- `scripts/sft.sh`
- `scripts/rl.sh`
- `scripts/evaluate.sh`
- `scripts/preprocess_amazon18.sh`
- `scripts/preprocess_amazon23.sh`
- `scripts/convert_dataset.sh`
- `scripts/text2emb.sh`

杩欎簺鑴氭湰鍙仛涓€灞傞潪甯歌杽鐨勫懡浠ゅ寘瑁呫€?
---

## 4. 閰嶇疆鏂囦欢缁勭粐

鏂伴厤缃綅浜庯細

- `configs/stages`

褰撳墠宸叉湁绀轰緥閰嶇疆锛?
- `configs/stages/preprocess/amazon18.yaml`
- `configs/stages/preprocess/amazon23.yaml`
- `configs/stages/embed/default.yaml`
- `configs/stages/sid/rqvae_train.yaml`
- `configs/stages/sid/rqvae_generate.yaml`
- `configs/stages/convert/default.yaml`
- `configs/stages/sft/default.yaml`
- `configs/stages/rl/default.yaml`
- `configs/stages/evaluate/default.yaml`

閰嶇疆椋庢牸鏄細

- YAML 璐熻矗涓婚厤缃?- CLI 璐熻矗瑕嗙洊涓埆瀛楁

渚嬪锛?
```bash
python -m minionerec.cli.main sft \
  --config configs/stages/sft/default.yaml \
  training.seed=7 \
  output.output_dir=./output/sft_debug
```

---

## 5. 涓绘祦绋嬮€昏緫閾炬潯

杩欎竴鑺傝鏄庡悇闃舵鐨勮緭鍏ャ€佸鐞嗛€昏緫鍜岃緭鍑恒€?
## 5.1 `preprocess`

鐩爣锛?
- 鎶婂師濮?Amazon review / metadata 鏁版嵁澶勭悊鎴愯缁冨墠鐨勫師瀛愭暟鎹牸寮?
瀹炵幇浣嶇疆锛?
- `src/minionerec/preprocess/amazon18.py`
- `src/minionerec/preprocess/amazon23.py`

涓昏閫昏緫锛?
- 璇诲彇鍘熷 metadata 鍜?reviews
- 鍋氭椂闂磋寖鍥磋繃婊?- 鍋?k-core 杩囨护
- 灏嗙敤鎴蜂氦浜掓暣鐞嗘垚鎸夋椂闂存帓搴忕殑搴忓垪
- 鍒囧垎涓?`train / valid / test`
- 鍐欏嚭锛?  - `*.train.inter`
  - `*.valid.inter`
  - `*.test.inter`
  - `*.item.json`

杈撳叆锛?
- 鍘熷 metadata 鏂囦欢
- 鍘熷 reviews 鏂囦欢

杈撳嚭锛?
- 鏌愪釜鏁版嵁闆嗙洰褰曚笅鐨?`inter` 鏂囦欢鍜?`item.json`

## 5.2 `embed`

鐩爣锛?
- 鎶?item 鐨勬枃鏈壒寰佺紪鐮佹垚 embedding

瀹炵幇浣嶇疆锛?
- `src/minionerec/sid/text2emb.py`

涓昏閫昏緫锛?
- 璇诲彇 `item.json`
- 浠庢瘡涓?item 鎻愬彇 `title + description`
- 鐢?embedding 妯″瀷缂栫爜
- 杈撳嚭 `.emb-xxx.npy`

杈撳叆锛?
- `item.json`
- embedding 妯″瀷璺緞

杈撳嚭锛?
- `*.emb-qwen-td.npy`

## 5.3 `sid-train`

鐩爣锛?
- 鐢?embedding 璁粌绂绘暎 SID 鏋勯€犲櫒

瀹炵幇浣嶇疆锛?
- `src/minionerec/sid/quantizers/rqvae.py`
- `src/minionerec/sid/quantizers/rqkmeans_faiss.py`
- `src/minionerec/sid/quantizers/rqkmeans_constrained.py`
- `src/minionerec/sid/quantizers/rqkmeans_plus.py`

涓昏閫昏緫锛?
- 杈撳叆 item embedding
- 璁粌澶氬眰娈嬪樊閲忓寲妯″瀷
- 杈撳嚭閲忓寲鍣?checkpoint / codebook

杈撳叆锛?
- `*.emb-qwen-td.npy`

杈撳嚭锛?
- quantizer checkpoint
- codebook / 涓棿浜х墿

## 5.4 `sid-generate`

鐩爣锛?
- 鎶婇噺鍖栨ā鍨嬭緭鍑鸿浆鎹㈡垚姝ｅ紡鐨?SID 绱㈠紩鏂囦欢

瀹炵幇浣嶇疆锛?
- `src/minionerec/sid/generate/rqvae_indices.py`
- `src/minionerec/sid/generate/rqkmeans_plus_indices.py`

涓昏閫昏緫锛?
- 鍔犺浇璁粌濂界殑閲忓寲鍣?- 瀵规瘡涓?item 鐢熸垚澶氬眰 code path
- 杞垚 token 褰㈠紡鐨?SID锛屼緥濡?`<a_42><b_17><c_203>`
- 澶勭悊纰版挒
- 鍐欏嚭 `index.json`

杈撳叆锛?
- item embedding
- quantizer checkpoint

杈撳嚭锛?
- `*.index.json`

## 5.5 `convert`

鐩爣锛?
- 鎶?`inter + item.json + index.json` 杞崲鎴?SFT / RL 浣跨敤鐨?CSV 涓?`info.txt`

瀹炵幇浣嶇疆锛?
- `src/minionerec/data/convert.py`
- `src/minionerec/cli/convert.py`

涓昏閫昏緫锛?
- 璇诲彇 `*.train.inter / *.valid.inter / *.test.inter`
- 璇诲彇 `item.json`
- 璇诲彇 `index.json`
- 灏?item id 鏇挎崲涓?semantic ID
- 鐢熸垚锛?  - `train/*.csv`
  - `valid/*.csv`
  - `test/*.csv`
  - `info/*.txt`

`info.txt` 鐨勪綔鐢ㄥ緢鍏抽敭锛?
- 鎻愪緵 `SID -> title -> item_id` 鏄犲皠
- 缁?RL 鍜?evaluate 鏋勫缓绾︽潫瑙ｇ爜鍓嶇紑琛?
杈撳叆锛?
- `item.json`
- `index.json`
- `.inter`

杈撳嚭锛?
- CSV
- `info.txt`

## 5.6 `sft`

鐩爣锛?
- 璁?LLM 瀛︿細鍦ㄤ富绾夸换鍔′笂鐢熸垚涓嬩竴涓?item SID

瀹炵幇浣嶇疆锛?
- `src/minionerec/training/sft/pipeline.py`
- `src/minionerec/training/sft/token_extension.py`
- 鏁版嵁闆嗗疄鐜颁綅浜庯細
  - `src/minionerec/data/datasets/sft.py`

涓荤嚎浣跨敤鐨?3 涓暟鎹泦锛?
- `SidSFTDataset`
  - 鍘嗗彶 SID -> 涓嬩竴涓?SID
- `SidItemFeatDataset`
  - SID <-> 鏍囬
- `FusionSeqRecDataset`
  - 鍘嗗彶 SID -> 涓嬩竴鐗╁搧鏍囬

涓昏閫昏緫锛?
- 鍔犺浇 base model
- 鎵╁睍 tokenizer锛屽姞鍏?SID token
- 缁勫悎涓荤嚎 SFT 鏁版嵁闆?- 浣跨敤 HuggingFace Trainer 璁粌
- 杈撳嚭 checkpoint 鍜?`final_checkpoint`

杈撳叆锛?
- base model
- train CSV
- valid CSV
- `index.json`
- `item.json`

杈撳嚭锛?
- `output/.../final_checkpoint`

## 5.7 `rl`

鐩爣锛?
- 鍦?SFT 妯″瀷鍩虹涓婂仛鎺ㄨ崘瀵煎悜 RL 璁粌

瀹炵幇浣嶇疆锛?
- `src/minionerec/training/rl/pipeline.py`
- `src/minionerec/training/rl/rewards.py`
- `src/minionerec/training/rl/trainer.py`

涓荤嚎浣跨敤鐨?3 涓暟鎹泦锛?
- `SidDataset`
  - SID history -> next SID
- `RLTitle2SidDataset`
  - title / description -> SID
- `RLSeqTitle2SidDataset`
  - title sequence -> SID

涓昏閫昏緫锛?
- 鍔犺浇 SFT checkpoint
- 璇诲彇 `info.txt`
- 鏋勫缓 `prompt2history` / `history2target`
- 鏋勫缓 reward function
- 浣跨敤 `ReReTrainer` 鍋氱害鏉熺敓鎴?+ GRPO 璁粌

褰撳墠涓荤嚎 reward 鏀寔锛?
- `rule`
- `ranking`
- `semantic`

杈撳叆锛?
- SFT checkpoint
- train / valid CSV
- `index.json`
- `item.json`
- `info.txt`

杈撳嚭锛?
- RL checkpoint
- `final_checkpoint`

## 5.8 `evaluate`

鐩爣锛?
- 瀵规ā鍨嬪仛鍙楃害鏉?Top-K 鎺ㄨ崘璇勪及

瀹炵幇浣嶇疆锛?
- `src/minionerec/evaluation/pipeline.py`
- `src/minionerec/evaluation/constrained_decoding.py`
- `src/minionerec/evaluation/split_merge.py`
- `src/minionerec/evaluation/metrics.py`

涓昏閫昏緫锛?
- 浠?`info.txt` 鏋勫缓鍚堟硶 SID 鍓嶇紑琛?- 鍦ㄧ敓鎴愭椂鍙厑璁歌緭鍑哄悎娉?SID
- 瀵规祴璇曢泦鐢熸垚 beam candidates
- 鍐欏嚭棰勬祴 JSON
- 鍐嶇敤 merge / calc 缁熻鎸囨爣

杈撳叆锛?
- 妯″瀷 checkpoint
- test CSV
- `info.txt`

杈撳嚭锛?
- 棰勬祴缁撴灉 JSON
- HR / NDCG / CC 缁熻

---

## 6. 褰撳墠鍚勯樁娈典箣闂寸殑渚濊禆鍏崇郴

鎺ㄨ崘浠庡墠鍒板悗渚濇鎵ц锛?
```text
1. preprocess
   -> 鐢熸垚 .inter 鍜?item.json

2. embed
   -> 璇诲彇 item.json锛岀敓鎴?item embedding

3. sid-train
   -> 璇诲彇 embedding锛岃缁冮噺鍖栧櫒

4. sid-generate
   -> 璇诲彇閲忓寲鍣?checkpoint锛岀敓鎴?index.json

5. convert
   -> 璇诲彇 .inter + item.json + index.json
   -> 鐢熸垚 train/valid/test CSV + info.txt

6. sft
   -> 璇诲彇 CSV + item.json + index.json
   -> 杈撳嚭 final_checkpoint

7. rl
   -> 璇诲彇 SFT checkpoint + CSV + info.txt
   -> 杈撳嚭 RL final_checkpoint

8. evaluate
   -> 璇诲彇 checkpoint + test CSV + info.txt
   -> 杈撳嚭缁撴灉 JSON 鍜屾寚鏍?```

鏇寸洿瑙傚湴璇达細

- `item.json` 鏄墍鏈変笅娓哥殑鍩虹鍏冧俊鎭?- `index.json` 鏄?SID 涓婚敭
- `info.txt` 鏄害鏉熻В鐮佺殑鏍稿績绱㈠紩
- `SFT final_checkpoint` 鏄?RL 鐨勪笂娓歌緭鍏?- `RL final_checkpoint` 鎴?`SFT final_checkpoint` 閮藉彲浠ヤ綔涓?evaluate 鐨勬ā鍨嬭緭鍏?
---

## 7. 鎺ㄨ崘鎿嶄綔鏂瑰紡

## 7.1 鎺ㄨ崘鍦ㄨ繙绋嬫湇鍔″櫒涓婃搷浣?
鐢变簬瀹屾暣璁粌渚濊禆锛?
- `torch`
- `transformers`
- `trl`
- `deepspeed`
- 澶氬崱 / 澶ф樉瀛樼幆澧?
鎵€浠ユ帹鑽愮殑瀹為檯浣跨敤鏂瑰紡鏄細

- 鏈湴鍋氱粨鏋勬鏌ャ€侀厤缃鏌ャ€佸绾︽祴璇?- 杩滅▼鏈嶅姟鍣ㄥ仛姝ｅ紡璁粌涓庤瘎浼?
## 7.2 鏈湴鍙仛鐨勪簨鎯?
鏈湴鎺ㄨ崘鍋氾細

- 鐪嬮厤缃槸鍚︽纭?- 璺?CLI help
- 璺戝崟鍏冩祴璇?- 璺?convert 濂戠害娴嬭瘯

绀轰緥锛?
```bash
python -m minionerec.cli.main --help
python -m unittest discover -s tests/unit -v
```

## 7.3 杩滅▼鏈嶅姟鍣ㄦ帹鑽愭祦绋?
### 姝ラ 1锛氱幆澧冨畨瑁?
寤鸿锛?
```bash
pip install -r requirements.txt
pip install -e .
```

璇存槑锛?
- `pip install -e .` 涓嶆槸寮哄埗锛屼絾鎺ㄨ崘
- 鍗充娇涓嶅畨瑁咃紝褰撳墠浠撳簱涔熸敮鎸佺洿鎺?`python -m minionerec.cli.main ...`

### 姝ラ 2锛氶澶勭悊

Amazon18 绀轰緥锛?
```bash
python -m minionerec.cli.main preprocess \
  --config configs/stages/preprocess/amazon18.yaml
```

### 姝ラ 3锛氱敓鎴?embedding

```bash
python -m minionerec.cli.main embed \
  --config configs/stages/embed/default.yaml
```

### 姝ラ 4锛氳缁?SID 閲忓寲鍣?
```bash
python -m minionerec.cli.main sid-train \
  --config configs/stages/sid/rqvae_train.yaml
```

### 姝ラ 5锛氱敓鎴?SID 绱㈠紩

杩欎竴闃舵閫氬父闇€瑕侀澶栬鐩栧弬鏁帮細

```bash
python -m minionerec.cli.main sid-generate \
  ckpt_path=./output/rqvae_Industrial_and_Scientific/xxx/best_collision_model.pth \
  output_file=./data/Amazon/index/Industrial_and_Scientific.index.json
```

### 姝ラ 6锛氳浆鎹㈣缁冩暟鎹?
```bash
python -m minionerec.cli.main convert \
  --config configs/stages/convert/default.yaml
```

### 姝ラ 7锛歋FT

```bash
python -m minionerec.cli.main sft \
  --config configs/stages/sft/default.yaml
```

### 姝ラ 8锛歊L

```bash
python -m minionerec.cli.main rl \
  --config configs/stages/rl/default.yaml
```

### 姝ラ 9锛歟valuate

```bash
python -m minionerec.cli.main evaluate \
  --config configs/stages/evaluate/default.yaml
```

---

## 8. 鍏煎鎿嶄綔鏂瑰紡

濡傛灉浣犵殑鏈嶅姟鍣ㄨ剼鏈巻鍙蹭笂涓€鐩寸敤鏃у叆鍙ｏ紝涔熷彲浠ユ殏鏃剁户缁繖鏍疯皟鐢細

```bash
python sft.py ...
python rl.py ...
python evaluate.py ...
python convert_dataset.py ...
```

鎴栵細

```bash
bash sft.sh
bash rl.sh
bash evaluate.sh
```

杩欎簺鍏ュ彛鐩墠浠嶇劧鍙敤锛屼絾鎺ㄨ崘閫愭杩佺Щ鍒帮細

```bash
python -m minionerec.cli.main ...
```

---

## 9. 褰撳墠闇€瑕佹敞鎰忕殑闄愬埗

褰撳墠閲嶆瀯鐗堝凡缁忓畬鎴愪富閾捐矾缁勭粐锛屼絾鏈夊嚑鐐归渶瑕佹槑纭細

1. 鏈湴鐜濡傛灉娌℃湁瀹夎 `torch` 绛夎缁冧緷璧栵紝涓嶈兘鐩存帴璺?SFT / RL / evaluate 姝ｅ紡娴佺▼銆?2. 褰撳墠 `configs/stages/*` 閲岀殑 YAML 鏄ず渚嬮厤缃紝姝ｅ紡璁粌鍓嶉渶瑕佹牴鎹湇鍔″櫒璺緞鍜屾ā鍨嬭矾寰勮皟鏁淬€?3. `archive/` 涓殑浠ｇ爜涓嶅睘浜庡綋鍓嶄富閾捐矾锛屼笉搴斾綔涓烘寮忚缁冨叆鍙ｃ€?4. `rq/` 鍜屾牴鐩綍鏃ц剼鏈粛鐒朵繚鐣欙紝鏄负浜嗗吋瀹癸紝涓嶆槸鏂扮殑涓诲叆鍙ｃ€?5. 褰撳墠鏈€鍙潬鐨勪娇鐢ㄦ柟寮忔槸锛?   - 鏈湴鍋氭鏌?   - 鏈嶅姟鍣ㄥ仛璁粌

---

## 10. 鎺ㄨ崘浣犵幇鍦ㄦ€庝箞鐢?
濡傛灉浣犳帴涓嬫潵瑕佺户缁紑鍙戞垨璁粌锛屾帹鑽愭寜涓嬮潰鏂瑰紡宸ヤ綔锛?
### 寮€鍙戞椂

- 涓昏鐪?`src/minionerec`
- 浼樺厛鏀规柊涓荤嚎浠ｇ爜
- 闈炲繀瑕佷笉瑕佺户缁湪鏍圭洰褰曟棫瀹炵幇涓婂姞鍔熻兘

### 璋冭瘯鏃?
- 鐢?YAML 閰嶇疆 + CLI override
- 灏忔牱鏈厛璺?`convert`
- 鍐嶅湪鏈嶅姟鍣ㄤ笂璺戝皬鏍锋湰 `sft`
- 鐒跺悗鍐嶆帴 `rl`

### 姝ｅ紡璁粌鏃?
- 鏈嶅姟鍣ㄤ笂缁熶竴浣跨敤锛?
```bash
python -m minionerec.cli.main <stage> --config <yaml>
```

---

## 11. 鐩稿叧鏂囦欢绱㈠紩

### 鏂颁富鍏ュ彛

- `src/minionerec/cli/main.py`

### SFT

- `src/minionerec/training/sft/pipeline.py`
- `src/minionerec/data/datasets/sft.py`

### RL

- `src/minionerec/training/rl/pipeline.py`
- `src/minionerec/training/rl/trainer.py`
- `src/minionerec/training/rl/rewards.py`

### Evaluate

- `src/minionerec/evaluation/pipeline.py`
- `src/minionerec/evaluation/constrained_decoding.py`
- `src/minionerec/evaluation/metrics.py`

### SID

- `src/minionerec/sid/text2emb.py`
- `src/minionerec/sid/quantizers/rqvae.py`
- `src/minionerec/sid/generate/rqvae_indices.py`

### Convert

- `src/minionerec/data/convert.py`

### 閰嶇疆

- `configs/stages`

### 鍏煎灞?
- `src/minionerec/compat/legacy_cli.py`

---

## 12. 涓€鍙ヨ瘽鎬荤粨

鐜板湪鐨勯」鐩凡缁忔槸鈥滃弻灞傜粨鏋勨€濓細

- **鐪熸鐨勪富绾垮疄鐜?*鍦?`src/minionerec`
- **鏃ф枃浠跺拰鏃ц剼鏈?*鍙槸鍏煎鍏ュ彛

浣犱互鍚庡簲璇ユ妸锛?
- 璇讳唬鐮?- 鏀逛唬鐮?- 閰嶇疆璁粌
- 缁勭粐涓绘祦绋?
閮藉敖閲忓洿缁?`src/minionerec + configs/stages + python -m minionerec.cli.main` 鏉ヨ繘琛屻€?
