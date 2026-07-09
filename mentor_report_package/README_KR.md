# AICC 음성 파이프라인 — 도구 테스트 결과 보고

상태: 예비 도구/파이프라인 검증 결과이며, 최종 연구 결과가 아닙니다.

## 이 문서는 무엇인가

이 패키지는 모듈형 AICC 음성 파이프라인에 대한 베이스라인 도구 셋업 테스트를
정리한 것입니다. 이번 테스트의 목적은 파이프라인 구성 요소들이 엔드투엔드로
정상 작동하는지 확인하고, 약한 로컬 LLM 베이스라인(Llama 3.2)이 어디에서
한계를 보이는지 초기 단계에서 있는 그대로 살펴보는 것이었습니다. 최종 제안
아키텍처를 검증하는 자료가 아니며, 성능을 주장하는 자료도 아닙니다.

## 패키지 구성

```
README_EN.md / README_KR.md                             영어/한국어 안내 문서 (본 파일)
METHODS_EN.md / METHODS_KR.md                            무엇을, 어떻게 테스트했는지
RESULTS_INTERPRETATION_EN.md / _KR.md                    수치가 의미하는 바
LIMITATIONS_AND_NEXT_STEPS_EN.md / _KR.md                알려진 한계와 다음 단계
results/
  component_results.csv                                  STT/TTS/LLM 개별 컴포넌트 테스트 결과 (1행 = 1회 테스트)
  pipeline_results.csv                                    전체 파이프라인 실행 결과 (총 48회 실행, 1행 = 1회 실행)
  summary.csv                                              파이프라인 조합별 지연 시간/오류 집계
outputs_for_inspection/
  transcripts/                                             수동 확인용 STT 전사 결과
  llm_outputs/                                              수동 확인용 Ollama 원본/파싱된 JSON 출력
config/
  routes.json                                              LLM 라우팅 판단에 사용된 경로 트리
  manifest.csv                                              배치 테스트에 사용된 시나리오 매니페스트
code_snapshot/
  src/, test_stt.py, test_tts.py, test_llm.py,
  run_pipeline.py, run_batch.py, compare_results.py,
  requirements.txt                                          테스트를 이해하거나 재현하는 데 필요한 코드
```

## 테스트 진행 방식

1. 각 컴포넌트(Google STT, 로컬 faster-whisper STT, Google TTS, 로컬 Piper
   TTS, Ollama/Llama 3.2)를 먼저 개별적으로 테스트했습니다.
2. `config/manifest.csv`에 정의된 시나리오들을 `run_batch.py`를 통해 4가지
   STT+LLM+TTS 파이프라인 조합 전체에 대해 실행했습니다.
3. 각 실행의 지연 시간, 전사 결과, 라우팅 판단, 응답 텍스트를
   `results/pipeline_results.csv`에 자동으로 기록했습니다.
4. `compare_results.py`가 이 기록들을 집계하여 `results/summary.csv`를
   생성했습니다.

## 먼저 확인하면 좋은 파일

1. **RESULTS_INTERPRETATION_KR.md** — 실제 해석 결과를 쉬운 말로 정리한 문서.
2. **results/summary.csv** — 해당 해석의 근거가 되는 지연 시간/오류 수치.
3. **LIMITATIONS_AND_NEXT_STEPS_KR.md** — 이번 결과가 왜 베이스라인일 뿐 결론이
   아닌지에 대한 설명.
4. **outputs_for_inspection/** — 모델의 실제 출력을 직접 보고 싶다면 전사 결과와
   LLM 출력 몇 개를 직접 확인해 보세요.
5. **results/pipeline_results.csv** — 특정 시나리오를 더 깊이 들여다보고 싶을
   때 참고할 전체 행 단위 데이터.

## 의도적으로 제외한 항목

- 생성된 응답 음성 파일은 이 패키지에 **기본적으로 포함되지 않습니다**
  (zip 용량이 크게 늘어나고, 결과 검토에는 필요하지 않기 때문입니다). 별도의
  오디오 샘플 패키지가 필요하면 말씀해 주세요.
- 인증 정보(credentials), `.venv/`, 다운로드된 모델 파일(`models/`),
  `__pycache__/`, `.DS_Store`는 포함되지 않았습니다.

## 한 줄 요약

파이프라인은 엔드투엔드로 정상 작동하며 로그도 깔끔하게 기록됩니다. 다만
로컬 LLM 베이스라인(Llama 3.2)은 정확한 경로 스키마 준수 측면에서 아직
신뢰할 만한 수준이 아니며, 이는 첫 베이스라인 단계에서 예상되는 결과로서
다음 아키텍처 연구 단계의 동기가 되는 것이지 최종 결과가 아닙니다.
