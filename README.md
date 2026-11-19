# MahjongLM 2024 타패 리뷰

MahjongLM의 2024년 마작 로그에서 타패 결정을 추출하고, LLM으로 후보 패 또는 실제 타패를 평가합니다.

## 준비

- Python 3.12 이상, `uv`
- 데이터셋 접근 권한과 Hugging Face 로그인

```powershell
uv sync
uvx --from huggingface_hub hf auth login
uv run --with huggingface_hub python download_mahjonglm_2024.py
```

데이터를 이미 받았다면 토크나이저만 받을 수 있습니다.

```powershell
uv run --with huggingface_hub python download_mahjonglm_2024.py --tokenizer-only
```

## 결정 데이터 만들기

```powershell
uv run python mahjonglm_decisions.py --limit 100
```

결과는 `data/mahjonglm_2024_decisions.jsonl`에 저장됩니다. `--limit`을 생략하면 전체 데이터를 처리합니다. 각 행은 한 플레이어의 타패 결정이며 `decision_id`로 원본 국면과 연결됩니다.

## LLM 설정 및 평가

프로젝트 루트의 `.env`에 API 설정을 입력합니다.

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-6-luna
# OpenAI 호환 API를 사용할 때만 설정
# OPENAI_BASE_URL=https://your-server.example/v1
```

```powershell
# 손패의 서로 다른 후보 패 전부 평가
uv run python review_all_discards.py --row 0

# 실제로 버린 패 평가
uv run python review_actual_discard.py --row 0

# 후보 점수 합계가 100이 되도록 LLM에 요청
uv run python review_all_discards.py --row 0 --sum-scores-to-100
```

기본 결과는 `data/llm_reviews/row-<행 번호>-all.json` 또는 `row-<행 번호>-actual.json`에 저장됩니다. 결과에는 패 매수, 한글 패 이름, 점수, 이유, 본인 후로와 식별자가 포함됩니다. 이미 파일이 있으면 덮어쓰지 않고 건너뜁니다.

`--row`는 0부터 시작하는 행 번호이며, `--limit`은 연속 처리할 행 수입니다. 예: `--row 100 --limit 10`. 동시 요청 수는 `--workers`로 조정할 수 있습니다.

## 국면 조회

결정 ID로 해당 국의 공개 정보를 조회하려면 먼저 SQLite DB를 만듭니다.

```powershell
uv run python build_round_state_db.py
uv run python lookup_round_state.py "2024091422gm-00a9-0000-f8cb05f4:0:0:1"
```

조회 결과에는 장풍·국·도라, 플레이어별 버림패 순서·후로·리치 정보가 포함됩니다. 이 DB는 제3자가 볼 수 있는 공개 정보만 저장하며 손패와 실제 타패 정답은 넣지 않습니다. DB를 다시 만들려면 `--replace`를 추가합니다.

## 참고

- LLM은 현재 플레이어의 손패와 공개된 버림패·후로·리치 등을 평가합니다.
- `decision_id` 형식은 `game_id:round_index:actor_seat:actor_discard_number`입니다.
- API 타임아웃은 기본 600초입니다. 필요하면 `--timeout`으로 초 단위 조정이 가능합니다.
