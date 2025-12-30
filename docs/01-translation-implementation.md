# リアルタイム英日翻訳機能 実装計画書

**作成日**: 2025-12-31
**ステータス**: 計画中
**対象**: Amazon Nova 2 Sonic リアルタイム英語文字起こしアプリ

---

## 1. 目的と概要

### 1.1 目的

英語の文字起こし結果をリアルタイムで日本語に翻訳し、ユーザーに同時表示する機能を追加する。

### 1.2 要件

- **低遅延**: リアルタイム文字起こしの体験を損なわない翻訳速度
- **高品質**: 自然で読みやすい日本語翻訳
- **UI/UX**: 英語と日本語を見やすく同時表示
- **コスト効率**: 長時間利用でもコストが抑えられる設計

---

## 2. 技術選定

### 2.1 選定結果: Amazon Bedrock Claude Haiku 4.5

**選定理由**:

- ✅ **最新モデル**: Claude Haiku 4.5（2025年10月リリース）
- ✅ **高性能**: Sonnet 4 に匹敵する性能を低コスト・高速で実現
- ✅ **高品質**: 文脈理解に優れ、自然な翻訳が可能
- ✅ **ストリーミング対応**: 段階的に翻訳結果を取得可能（体感速度向上）
- ✅ **柔軟性**: プロンプトで翻訳のトーンや文体を調整可能
- ✅ **既存インフラ**: Bedrock クライアントを再利用可能

**代替案との比較**:

| 項目 | Claude Haiku 4.5 | Amazon Translate |
|------|----------------|------------------|
| レイテンシ | 200-500ms | 50-100ms ◎ |
| 翻訳品質 | ◎ 非常に高い | ○ 高い |
| コスト | ○ 中程度 | ◎ 低い |
| ストリーミング | ◎ 対応 | ○ 対応 |
| 実装複雑度 | ○ 中 | ◎ 低 |

**判断**: ユーザー要望により品質重視の Claude Haiku 4.5 を選択

**モデル ID**: `anthropic.claude-haiku-4-5-20251001-v1:0`

---

## 3. アーキテクチャ設計

### 3.1 システムフロー

```
ブラウザマイク
    ↓
[Nova 2 Sonic] → 英語文字起こし
    ↓
FastAPI サーバー
    ↓
[Claude 3 Haiku] → 英日翻訳（ストリーミング）
    ↓
WebSocket
    ↓
ブラウザUI（英語 + 日本語表示）
```

### 3.2 翻訳タイミング

**final（確定）テキストのみ翻訳**:
- Nova 2 Sonic から `final` イベントを受信時
- Claude 3 Haiku にストリーミング翻訳リクエスト
- 翻訳結果を逐次ブラウザに送信

**partial（部分）テキストは翻訳しない**:
- レイテンシとコストの観点から見送り
- 英語の部分結果のみ表示

### 3.3 データフロー

```python
# サーバー側
1. Nova 2 Sonic から final テキスト受信
   ↓
2. Claude 3 Haiku にストリーミング翻訳リクエスト
   ↓
3. ストリーミング翻訳結果を受信
   ↓
4. ブラウザに {"type": "translation", "text": "..."} を送信

# ブラウザ側
1. WebSocket で translation イベント受信
   ↓
2. 翻訳表示エリアに追加
   ↓
3. 自動スクロール
```

---

## 4. 実装項目

### 4.1 サーバー側 (main.py)

#### 4.1.1 Claude 3 Haiku クライアント初期化

```python
# Bedrock Runtime Client の再利用
# get_bedrock_client() は既存のまま使用可能
```

#### 4.1.2 翻訳関数の実装

```python
async def translate_text(client, text: str, websocket: WebSocket) -> None:
    """
    Claude Haiku 4.5 でストリーミング翻訳

    Args:
        client: Bedrock Runtime Client
        text: 翻訳対象の英語テキスト
        websocket: ブラウザへの送信用WebSocket
    """
    # プロンプト設計
    prompt = f"Translate the following English text to natural Japanese:\n\n{text}"

    # ストリーミングリクエスト
    response = await client.invoke_model_with_response_stream(
        modelId="anthropic.claude-haiku-4-5-20251001-v1:0",
        body={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    # ストリーミング結果を逐次送信
    translation_buffer = ""
    async for event in response['body']:
        chunk = json.loads(event['chunk']['bytes'])
        if chunk['type'] == 'content_block_delta':
            delta_text = chunk['delta']['text']
            translation_buffer += delta_text
            await websocket.send_text(json.dumps({
                "type": "translation",
                "text": delta_text
            }))
```

#### 4.1.3 read_outputs() 関数の修正

```python
async def read_outputs():
    # ...既存のコード...

    if "textOutput" in ev:
        text = ev["textOutput"].get("content", "")
        # ...既存の処理...

        # USERロールの final テキストを翻訳
        if session.current_role in ["USER", "ASSISTANT"]:
            stage_lower = (session.current_generation_stage or "FINAL").lower()

            # 既存の送信処理
            await websocket.send_text(json.dumps({"type": stage_lower, "text": text}))

            # final の場合のみ翻訳
            if stage_lower == "final":
                asyncio.create_task(translate_text(client, text, websocket))
```

### 4.2 フロントエンド (HTML/JavaScript)

#### 4.2.1 UI 構造の変更

```html
<!-- 英語表示エリア（既存） -->
<div id="transcript"></div>

<!-- 日本語翻訳表示エリア（新規） -->
<h3>Japanese Translation</h3>
<div id="translation"></div>
```

#### 4.2.2 JavaScript の変更

```javascript
// グローバル変数に追加
let finalTranslation = "";
let partialTranslation = "";

const translationDiv = document.getElementById("translation");

// 翻訳テキスト更新関数
function appendTranslation(text) {
  finalTranslation += text;
  updateTranslationView();
}

function updateTranslationView() {
  translationDiv.textContent = finalTranslation || "…";
  translationDiv.scrollTop = translationDiv.scrollHeight;
}

// WebSocket メッセージハンドラに追加
ws.onmessage = (event) => {
  try {
    const msg = JSON.parse(event.data);

    // ...既存の処理...

    // 翻訳結果の処理
    if (msg.type === "translation") {
      appendTranslation(msg.text);
    }
  } catch (e) {
    // ...
  }
};
```

### 4.3 スタイリング (CSS)

```css
#translation {
  white-space: pre-wrap;
  min-height: 360px;
  max-height: 500px;
  overflow-y: auto;
  background: #f9fafb;
  border: 1px solid #d9dde3;
  padding: 14px;
  border-radius: 10px;
  font-size: 16px;
  line-height: 1.5;
  margin-top: 20px;
}
```

---

## 5. 依存関係

### 5.1 必要なパッケージ

既存の依存関係のみで実装可能:
- `aws_sdk_bedrock_runtime`: Bedrock API 呼び出し（既存）
- `fastapi`: WebSocket 通信（既存）
- `python-dotenv`: 環境変数管理（既存）

**追加パッケージ**: なし

---

## 6. IAM 権限

### 6.1 必要な追加権限

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.nova-2-sonic-v1:0",
        "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"
      ]
    }
  ]
}
```

### 6.2 Model Access

AWS コンソール → Bedrock → Model access で以下を有効化:

- ✅ Amazon Nova 2 Sonic (既存)
- ✅ Anthropic Claude Haiku 4.5 (**新規追加が必要**)

---

## 7. パフォーマンス考慮事項

### 7.1 レイテンシ削減

**ストリーミング翻訳の活用**:
- Claude 3 Haiku のストリーミングレスポンスを使用
- 翻訳完了を待たずに、生成された文字から逐次表示
- 体感速度の向上（TTFT: Time To First Token を重視）

**非同期処理**:
```python
asyncio.create_task(translate_text(...))
```
- 翻訳をバックグラウンドタスクで実行
- 文字起こしの処理をブロックしない

### 7.2 コスト最適化

**final テキストのみ翻訳**:
- partial（部分結果）は翻訳しない
- トークン使用量を最小限に抑制

**プロンプトの簡潔化**:
```python
prompt = f"Translate to Japanese:\n\n{text}"
```
- 不要な説明文を削除
- 入力トークン数を削減

### 7.3 エラーハンドリング

```python
try:
    await translate_text(client, text, websocket)
except Exception as e:
    logger.error(f"Translation error: {e}")
    await websocket.send_text(json.dumps({
        "type": "translation_error",
        "error": str(e)
    }))
```

---

## 8. 実装手順

### 8.1 フェーズ1: サーバー側実装

1. ⬜ IAM 権限の確認・追加
2. ⬜ Bedrock Model Access で Claude Haiku 4.5 を有効化
3. ⬜ `translate_text()` 関数の実装
4. ⬜ `read_outputs()` 関数の修正（翻訳呼び出し）
5. ⬜ エラーハンドリングの追加
6. ⬜ ログ出力の追加

### 8.2 フェーズ2: フロントエンド実装

1. ⬜ HTML に翻訳表示エリアを追加
2. ⬜ CSS スタイリングの追加
3. ⬜ JavaScript 翻訳処理の実装
4. ⬜ WebSocket メッセージハンドラの拡張
5. ⬜ Clear ボタンに翻訳クリア処理を追加
6. ⬜ Download 機能に翻訳テキストを含める

### 8.3 フェーズ3: テスト・調整

1. ⬜ ローカル環境でのテスト
2. ⬜ 翻訳品質の確認
3. ⬜ レイテンシの測定・最適化
4. ⬜ エラーケースのテスト
5. ⬜ README の更新

### 8.4 フェーズ4: ドキュメント更新

1. ⬜ README にリアルタイム翻訳機能の説明を追加
2. ⬜ IAM 権限の追加手順を記載
3. ⬜ Model Access 設定手順を追加
4. ⬜ スクリーンショット更新（可能であれば）

---

## 9. 成功基準

### 9.1 機能要件

- ✅ 英語 final テキストが自動的に日本語翻訳される
- ✅ 翻訳結果がリアルタイムで表示される
- ✅ 英語と日本語が同時に画面に表示される
- ✅ 自動スクロールが正常に動作する

### 9.2 非機能要件

- ✅ 翻訳レイテンシ: 1秒以内（TTFT）
- ✅ 翻訳品質: 自然で読みやすい日本語
- ✅ エラー時も文字起こしは継続
- ✅ UI がレスポンシブで使いやすい

---

## 10. リスクと対策

### 10.1 リスク

| リスク | 影響 | 対策 |
|--------|------|------|
| Claude 3 Haiku のレイテンシが高い | ユーザー体験低下 | ストリーミングで体感速度向上 |
| Model Access が有効化されていない | 翻訳が動作しない | README に明記、エラーメッセージ改善 |
| 翻訳コストが高騰 | 運用コスト増 | final のみ翻訳、プロンプト簡潔化 |
| 長文の翻訳が途切れる | 翻訳品質低下 | max_tokens を適切に設定 |

### 10.2 フォールバック戦略

**翻訳失敗時**:
- エラーログを出力
- ブラウザにエラー通知（軽微な表示）
- 英語の文字起こしは継続

---

## 11. 将来の拡張性

### 11.1 オプション機能（将来検討）

- [ ] 翻訳エンジンの切り替え（Claude / Translate）
- [ ] 翻訳言語の選択（英→日以外）
- [ ] 翻訳の ON/OFF トグル
- [ ] 翻訳品質の設定（速度 vs 品質）
- [ ] 文字起こし + 翻訳の統合ダウンロード

---

## 12. 参考資料

### Claude Haiku 4.5 関連

- [Claude 4.5 Haiku by Anthropic now in Amazon Bedrock - AWS](https://aws.amazon.com/about-aws/whats-new/2025/10/claude-4-5-haiku-anthropic-amazon-bedrock/)
- [Claude by Anthropic - Models in Amazon Bedrock – AWS](https://aws.amazon.com/bedrock/anthropic/)
- [Anthropic's Claude 4.5 Haiku Now Available in Amazon Bedrock -- AWSInsider](https://awsinsider.net/articles/2025/10/22/anthropics-claude-4-5-haiku-now-available-in-amazon-bedrock.aspx)

### 一般ドキュメント

- [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Claude on Amazon Bedrock - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock)
- [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Bedrock Streaming Response](https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-model-streaming.html)

---

**承認者**: ユーザー確認後、実装開始
**実装者**: Claude Code
**レビュアー**: ユーザー
