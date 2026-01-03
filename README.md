# Amazon Nova 2 Sonic リアルタイム英語文字起こし / 英日翻訳 Webアプリ

Amazon Bedrock の Nova 2 Sonic を使ったブラウザマイク音声からのリアルタイム英語文字起こしに加え、Claude 4.5 Haiku による英日翻訳（ストリーミング）も同時表示するアプリケーションです。

## 概要

- ブラウザのマイクから音声を取得
- Bedrock の Nova 2 Sonic 双方向ストリーミングで英語文字起こし
- Bedrock の Claude 4.5 Haiku で英日翻訳（**final のみ** / ストリーミング）
- 英語（USER final）と日本語を **1発話ごとに紐づけて表示**（Aligned EN ↔ JA）
- リアルタイム表示とTXTファイル保存機能
- 8分のストリーム寿命制限を自動更新で回避
- **画面スリープ防止機能**（モバイル対応 - Wake Lock API）
- **自動スクロール機能**（長文でも最新の文字起こしを自動表示）
- **低遅延最適化**（100msフレーム送信、シンプルなプロンプト）

## 前提条件

- **Python 3.12+** （aws_sdk_bedrock_runtime が Python>=3.12 を要求）
- **uv** パッケージマネージャー
- AWS アカウントと以下の設定：
  - Amazon Bedrock で **Nova 2 Sonic** / **Claude 4.5 Haiku** の Model access が有効
  - 適切な IAM 権限（`bedrock:InvokeModelWithBidirectionalStream` / `bedrock:InvokeModelWithResponseStream` など）
  - 利用リージョン: `ap-northeast-1`（Tokyo）推奨

## AWS 側の設定

### 1. Bedrock: Model access の有効化

1. AWS コンソール → Amazon Bedrock
2. **Model access** で以下を "Access granted" にする
   - **Amazon Nova 2 Sonic**
   - **Anthropic Claude 4.5 Haiku**
3. Anthropic 系モデルは、初回利用前に **use case details**（利用目的の申請）提出が必要な場合があります  
   エラーに `Model use case details have not been submitted` と出たら、フォームを提出して **15分ほど待ってから**再試行してください

### 2. IAM: ユーザーと権限

開発用 IAM ユーザーに以下の権限を付与：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithBidirectionalStream",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.nova-2-sonic-v1:0",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"
      ]
    }
  ]
}
```

**補足**:
- 翻訳で inference profile（ID/ARN）を使う場合、IAM の `Resource` に inference profile の ARN も許可してください（または運用方針に応じて `Resource: "*"`）。

### 3. アクセスキーの作成

1. IAM → ユーザー → アクセスキーを作成
2. ユースケース: **ローカルコード**
3. 表示される値を `.env` ファイルに記録

## セットアップ

### 1. uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、シェルを再起動するか：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 2. リポジトリのクローン/ダウンロード

```bash
cd /path/to/your/workspace
# または git clone などでコードを取得
```

### 3. .env ファイルの作成

プロジェクトルートに `.env` を作成し、AWS認証情報を記入：

```env
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION=ap-northeast-1

# 翻訳設定（オプション）
# 注意: Claude 4.5 Haiku はリージョンによって on-demand throughput が使えず、
# inference profile の ID/ARN 指定が必要な場合があります
# 推奨（inference profile 例）:
TRANSLATION_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
# on-demand が使える場合の例:
# TRANSLATION_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
# on-demand が使えない場合のフォールバック（必要なら変更）:
# TRANSLATION_MODEL_ID_FALLBACK=us.anthropic.claude-haiku-4-5-20251001-v1:0
# TRANSLATION_MAX_TOKENS=400
# 翻訳のスロットリング対策（必要なら調整）:
# TRANSLATION_DEBOUNCE_SECONDS=0.4
# TRANSLATION_MIN_INTERVAL_SECONDS=0.6
# TRANSLATION_MAX_BATCH_CHARS=1200

# デバッグログを有効にする場合（オプション）
# LOG_LEVEL=DEBUG
```

**重要**: `.env` ファイルは `.gitignore` に含まれています。Git にコミットしないでください。

**デバッグモード**: 文字起こしがうまく動かない場合、`.env` に `LOG_LEVEL=DEBUG` を追加すると、Bedrockからの全イベントがログに表示されます。

### 4. 依存関係のインストール

```bash
uv sync
```

これで `.venv/` 仮想環境が自動作成され、依存パッケージがインストールされます。

## 実行方法

### ローカル実行

プロジェクトルートで以下を実行：

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

ブラウザで `http://localhost:8000` を開きます。

### プロキシ経由での実行（Cloudflare Access など）

リバースプロキシやCloudflare Access経由で公開する場合、以下のオプションを追加：

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers --env-file .env
```

**重要な注意点**:

#### 1. タイムアウト対策（必須）

- **Cloudflare Free**: WebSocket接続は100秒アイドルでタイムアウト
- 本アプリには**30秒間隔のキープアライブ機能**が実装済み
- 音声送信中は自動的にタイムアウトを回避しますが、長時間無音の場合でも接続は維持されます

#### 2. WebSocket サポート

- Cloudflare Access は WebSocket を完全サポート
- 最初のHTTPリクエストで認証後、自動的に WebSocket にアップグレード
- バイナリデータ（PCM音声）の送信も問題なく動作

#### 3. SSL/TLS 終端

- プロキシでHTTPSを終端する場合、`--proxy-headers` オプションにより：
  - `X-Forwarded-For` ヘッダーから正しいクライアントIPを取得
  - `X-Forwarded-Proto` ヘッダーでプロトコル（https）を正しく認識

#### 4. 動作状態の可視化

- ブラウザUIに3つのステータスインジケーターを表示：
  - **WebSocket**: 接続状態（緑 = 接続中、赤 = エラー）
  - **Audio**: マイク音声の受信状態
  - **AWS**: Bedrock との通信状態
- サーバー側ログでも詳細な状態を確認可能

### 高パフォーマンス実行（本番環境向け）

低遅延・高スループットが必要な場合、以下のオプションを追加して実行：

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools --ws websockets --env-file .env
```

**オプション説明:**

- `--loop uvloop`: 高速な非同期イベントループ（Cython製、標準asyncioより2-4倍高速）
- `--http httptools`: 高速なHTTPパーサー（Cython製）
- `--ws websockets`: 高速なWebSocketライブラリ

**複数クライアント対応（マルチワーカー）:**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop --http httptools --ws websockets --env-file .env
```

**注意**: `--reload` と `--workers` は併用できません。開発時は `--reload`、本番環境では `--workers` を使用してください。

**追加パッケージのインストール:**

上記オプションを使用するには、追加パッケージが必要です：

```bash
uv add uvloop httptools websockets
```

## 使い方

1. ブラウザで `http://localhost:8000` にアクセス
2. **Start** ボタンをクリック
3. マイク権限を許可
4. 英語で話すと、リアルタイムで英語文字起こしが表示されます（partial/final）
5. **final（確定）** テキスト（USERロール）のみ、日本語翻訳が下部にストリーミング表示されます
6. **Aligned EN ↔ JA** で、英語（発話）と日本語（翻訳）が行単位で対応表示されます（スマホ幅では自動的に縦積み）
6. **Download TXT** で英語文字起こし + 日本語翻訳をテキストファイルとして保存
7. **Clear** で文字起こし/翻訳の両方をクリア
8. **Stop** で録音を停止

### セッション自動更新

Bedrock のストリーム寿命（8分）を回避するため、7分45秒ごとに自動でセッションが更新されます。画面に `Session renewed to avoid the 8-minute limit.` と表示されますが、文字起こしは継続されます。

## モバイル対応機能

### 画面スリープ防止（Wake Lock API）

スマートフォンやタブレットで長時間使用する際、画面が自動的にオフになるのを防ぎます。

- **対応ブラウザ**:
  - iOS Safari 16.4+ / iPadOS 16.4+
  - Android Chrome / Edge
  - デスクトップ Chrome / Edge
- **動作**: Startボタンを押すと自動的に有効化、Stopボタンで解放
- **非対応ブラウザ**: エラーにならず、単純にスリープ防止が無効になるだけです

### 自動スクロール

文字起こしが長くなっても、常に最新のテキストが画面に表示されるよう自動スクロールします。

- 文字起こし表示エリアは最大高さ500pxで、それを超えるとスクロール可能になります
- 新しいテキストが追加されるたびに自動的に最下部までスクロール

## パフォーマンス最適化

本アプリは低遅延なリアルタイム文字起こしのために以下の最適化を実装しています：

### 音声フレームサイズ

- **設定値**: 100ms（1600サンプル at 16kHz）
- **効果**: ネットワークオーバーヘッドの削減とレスポンス速度のバランスを最適化

### システムプロンプト

- **設定**: シンプルで直接的な指示文
- **効果**: モデルの初期処理時間を短縮（Time to First Token の改善）

### ブラウザ側音声処理（高パフォーマンス）

**AudioWorklet 実装:**

- **最新ブラウザ**: AudioWorklet で音声処理をUIスレッドから分離（高パフォーマンス）
- **古いブラウザ**: ScriptProcessorNode に自動フォールバック（互換性確保）
- **効果**: UIブロック解消、GC削減、CPU負荷軽減

**リングバッファ最適化:**

- **固定サイズバッファ**: 8000サンプル（500ms分）で再割り当てを防止
- **フレーム送信用固定バッファ**: ArrayBuffer事前確保でGC削減
- **コピー削減**: DataViewへの直接書き込みで連結操作を最小限に
- **効果**: メモリ効率向上、CPU負荷軽減

**変換処理の最適化:**

- **Float32→Int16変換**: 事前確保配列を使用してGC削減
- **ダウンサンプリング**: インプレース処理で不要なメモリ割り当てを回避
- **効果**: CPU負荷軽減、メモリ効率向上

### サーバー側最適化

**キューの深さ削減:**

- **設定値**: 20フレーム（2秒分）
- **効果**: リアルタイム性を優先し、遅延の蓄積を防止
- **動作**: キューが満杯時は古いフレームを破棄（リアルタイム性優先）

### その他の最適化

- **発話終了検知**: `endpointingSensitivity: HIGH` で高速化
- **非圧縮PCM**: デコード処理を回避して遅延を最小化
- **WebSocket接続維持**: ハンドシェイクオーバーヘッドを削減
- **部分結果の活用**: 話している最中から文字起こしを表示（体感速度の向上）

## トラブルシューティング

### AccessDenied / Unauthorized エラー

- IAM ユーザーに `bedrock:InvokeModelWithBidirectionalStream` / `bedrock:InvokeModelWithResponseStream` 権限があるか確認
- Bedrock の Model access が許可されているか確認
- `.env` のキーが正しいか確認（空白や全角文字の混入に注意）

### 書き起こし中に `{ "interrupted": true }` が出てくる

- これは音声の本文ではなく、モデルが返すことがある **システム的なタグ**です
- 本アプリではこの形式の出力は表示/翻訳対象から除外しています

### リージョン不一致エラー

- `.env` の `AWS_REGION` が Nova 2 Sonic 提供リージョンか確認
- Tokyo リージョン: `ap-northeast-1`

### マイクが使えない

- `localhost` または HTTPS で開いているか確認（ブラウザの仕様で制限あり）
- OS 側のマイク権限を確認

### Import エラー

- Python 3.12 以上を使用しているか確認：
  ```bash
  uv run python --version
  ```

### 文字起こしが動作しない（音声は届いているが結果が0件）

音声フレームは受信されているのに文字起こしが表示されない場合：

1. **デバッグログを有効化**: `.env` に `LOG_LEVEL=DEBUG` を追加して再起動
2. **Bedrockからのイベントを確認**: ログに以下が表示されているか確認
   - `Content started: role=USER` → Bedrockが音声を認識
   - `Received textOutput: role=USER` → 文字起こし結果を受信
   - これらが表示されない場合、Bedrockが音声を処理していない可能性があります
3. **考えられる原因**:
   - 音声が小さすぎる/無音状態（マイクの音量を確認）
   - 英語以外の言語で話している（Nova 2 Sonicは英語専用）
   - ブラウザのマイク設定が正しくない
   - Bedrock側のエラー（ログに `error` イベントが表示されます）

### プロキシ経由での接続が切れる

- **WebSocket タイムアウト**: Cloudflare Free は100秒アイドルで切断されます
  - 本アプリは30秒間隔でキープアライブpingを送信（対策済み）
  - ブラウザのステータスインジケーターで接続状態を確認
- **サーバー側ログ確認**: ターミナルに接続状態の詳細ログが表示されます
  - `Opening Bedrock session...` → AWS接続開始
  - `✓ Bedrock session opened successfully` → 接続成功
  - `Audio receiving: N frames received` → 音声受信中
  - `Transcription #N` → 文字起こし成功

## 動作状態の確認方法

### ブラウザ側（UI）

画面上部に3つのステータスインジケーターが表示されます：

- **WebSocket** (緑点滅 = 正常):
  - Connected: サーバーとの接続が確立
  - Disconnected: 切断状態
  - Error (赤): 接続エラー

- **Audio** (緑点滅 = 正常):
  - Capturing: マイク音声を取得中
  - Receiving: 音声データをサーバーに送信中
  - Idle: 待機中

- **AWS** (緑点滅 = 正常):
  - Connected: Bedrock に接続済み
  - Transcribing: 文字起こし処理中
  - Error (赤): AWS接続エラー

### サーバー側（ログ）

ターミナルで以下のログが表示されます：

```log
2025-01-01 12:00:00 [INFO] WebSocket connection accepted
2025-01-01 12:00:01 [INFO] Opening Bedrock session...
2025-01-01 12:00:02 [INFO] ✓ Bedrock session opened successfully
2025-01-01 12:00:02 [INFO] ✓ Bedrock session initialized and ready
2025-01-01 12:00:05 [INFO] Audio receiving: 100 frames received
2025-01-01 12:00:10 [INFO] Transcription #1 (final): Hello world...
```

## セキュリティ注意事項

- `.env` ファイルは絶対に Git にコミットしない
- IAM アクセスキーは定期的にローテーション
- 本番環境では IAM ユーザーの長期キーではなく、IAM ロール、SSO、または Secrets Manager の使用を推奨

## 技術スタック

- **FastAPI**: Web フレームワーク
- **uvicorn**: ASGI サーバー
- **aws_sdk_bedrock_runtime**: Bedrock Python SDK
- **python-dotenv**: 環境変数管理
- **uv**: パッケージマネージャーと仮想環境管理

## ライセンス

このプロジェクトは個人利用・学習目的のサンプルコードです。
