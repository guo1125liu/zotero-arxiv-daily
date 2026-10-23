# 每日论文邮件：正式顶刊顶会与主题筛选

2026-10-23 起，本 fork 在原每日推送流程中加入强制筛选。发送时间、收发邮箱、中文摘要模型和最多 30 篇的上限由原配置继续管理。

## 两个条件必须同时满足

1. 主题是语言／多模态语言模型后训练，或视觉目标／缺陷／视觉异常检测。后训练包括 SFT、指令微调、偏好优化、RLHF／RLAIF、DPO、RLVR／GRPO、相关奖励模型等。泛 AI、流体、泛图像生成、纯预训练、后训练量化、网络入侵／文本异常检测不作为目标。摘要里只提到相关基线或下游任务，不足以入选。
2. DBLP 中有与标题和作者相符的正式发表记录，场所、记录类型与正式论文链接均通过白名单校验。不把 arXiv 评论里的“submitted”“accepted”或模型的判断当作正式发表证据。

白名单是本次订阅的明确范围，不表示覆盖所有优质期刊会议：

| 期刊 | 配置 ID |
| --- | --- |
| IEEE Transactions on Pattern Analysis and Machine Intelligence | TPAMI |
| International Journal of Computer Vision | IJCV |
| Journal of Machine Learning Research | JMLR |
| Artificial Intelligence | AIJ |
| IEEE Transactions on Image Processing | TIP |
| Nature Machine Intelligence | NMI |

| 正式主会 | 配置 ID |
| --- | --- |
| CVPR、ICCV、ECCV | CVPR、ICCV、ECCV |
| NeurIPS、ICML、ICLR | NEURIPS、ICML、ICLR |
| ACL、EMNLP | ACL、EMNLP |

Workshop、Findings、演示、学生研讨会、仅预印本不在范围内。未知白名单 ID 会报错，不会自动放宽。

## 运行顺序与无结果行为

arXiv 最新 RSS → 标题／摘要主题门槛 → DBLP 正式出处核验 → 合格论文全文提取 → 原 Zotero 相似度排序 → 原中文摘要 → 邮件。

邮件新增期刊／会议、年份、主题、正式论文页和 DBLP 核验链接。排序分数只表示与个人文献库的相关度，不等于期刊等级或论文质量。

没有合格论文时沿用 `send_empty: false`，不凑数。出处接口故障则该轮失败、不发邮件，并在 Actions 中明确记录核验失败；不会把接口故障说成“今天没有符合的论文”。Test 工作流同样执行筛选，调试上限作用于筛选之后。

## 配置与预演

个性化配置仍放在仓库 `CUSTOM_CONFIG` 变量；密钥只用环境变量引用。关键部分：

```yaml
curation:
  enabled: true
  backend: dblp_sparql
  allowed_venues: [TPAMI, IJCV, JMLR, AIJ, TIP, NMI, CVPR, ICCV, ECCV, NEURIPS, ICML, ICLR, ACL, EMNLP]
  timeout_seconds: 15
  max_retries: 1
  retry_backoff_seconds: 2
  max_hits: 10
```

每日和 Test 工作流设置 `REQUIRE_STRICT_CURATION=true`；若误关筛选，程序会中止。严格筛选目前仅支持 `executor.source: [arxiv]`，其他来源不会绕过门槛。候选分类移除 `physics.flu-dyn`；剩余 AI／CL／LG／CV／eess.IV 都必须经过两道门槛。

`Preview strict paper selection` 是手动只读预演工作流，无 Zotero／模型／SMTP 密钥，不发邮件、不下载全文，输出筛选报告 artifact。也可本地运行：

```powershell
uv run python scripts/preview_curation.py --config <custom.yaml路径> --report <新报告路径.json>
```

## 覆盖边界

- 发现入口仍是 arXiv 最新新稿 RSS，不是全部顶刊顶会的完整目录；已发表但没有出现在当前 RSS 的论文不会自动补推。
- DBLP 收录可能晚于出版社。无记录、作者不符、标题变化或无法验证链接时，宁可漏掉，也不以预印本补位。
- 实际使用 DBLP 官方 SPARQL 服务，该知识图谱每日与 DBLP 同步。普通搜索 API 可选，但本次环境返回验证页，不能作为已通过的真实核验。
- 主题是保守的可测试文字规则，仍可能漏掉用非常规术语描述的相关工作。它不是对论文研究质量的人工审稿。
- 没有新增定时任务；原 `Send emails daily` 的时间继续生效。

## 维护

本 fork 现在包含个性化代码。**不要再用 `gh repo sync --force` 或网页 Discard commits 覆盖它。** 更新上游应在分支合并、解决冲突、跑测试，并核对本筛选和工作流要求仍在。

默认本地测试：`uv run pytest`。包含模型下载测试的完整 CI：`uv run pytest -m ""`。真实网络核验和只读预演单独记录，离线单测不证明远端服务永远可用。

出处接口文档：[DBLP SPARQL](https://sparql.dblp.org/)、[官方知识图谱说明](https://github.com/dblp/kg/wiki/dblp-KG-Tutorial)。
