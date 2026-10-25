# 本机入库管道

代码：`E:\vector_library\vector_library`。数据库及 Python 环境：`F:\vector_library`。

## 放资料的位置

默认执行 `ingest` 会同时扫描以下三个位置，并写入所选管道的同一个知识库：

| 位置 | 用途 |
|---|---|
| `F:\vector_library\documents\inbox` | 本机资料，不通过 SMB 共享 |
| `F:\vector_library\shared\documents\inbox` | Windows / Linux 共享资料 |
| `F:\vector_library\shared\code` | Windows / Linux 共享代码 |

Linux 的共享根目录为 `smb://192.168.71.71/VectorFiles`。向量化任务在 Windows 主机执行。共享的 `exchange` 目录用于临时交换文件，默认不扫描；可通过 `ingest 路径` 单独处理。

配置项 `[project].sources_dirs` 指定多个默认来源；未配置时兼容原有 `sources_dir`。命令中明确指定文件或目录时，只处理该路径。扫描会对重叠目录中的同一路径去重；不同路径下的文件分别保留记录。未变化的论文会跳过。

支持 PDF（可提取文字）、Markdown、TXT、HTML、JSON、JSONL，以及 Python、JS/TS、C/C++、Java、Go、Rust、Shell、PowerShell、SQL、TOML、YAML 等代码文件。扫描版 PDF 尚不含 OCR。扫描时跳过隐藏目录、虚拟环境、依赖目录、数据库目录和已知凭据文件。

## 可用管道

| 管道 | 处理流程 | PostgreSQL 表前缀 |
|---|---|---|
| 本地 | 解析 → 切片 → Ollama BGE-M3 1024维 → pgvector | `knowledge_bge_m3_` |
| Qwen 云端增强 | 解析 → 切片 → qwen3.8-flash摘要/关键词 → 本地BGE-M3 1024维 → pgvector | `knowledge_qwen_bge_m3_` |
| 完整云端模型 | 解析 → 切片 → qwen3.8-flash摘要/关键词 → qwen3.7-text-embedding-flash 1024维 → pgvector | `knowledge_qwen37_enriched_` |
| 仅云端向量 | 解析 → 切片 → qwen3.7-text-embedding-flash 1024维 → pgvector | `knowledge_qwen37_flash_` |

Qwen 增强会把文档片段发送至你提供的阿里云接口，每个需更新的片段调用一次；模型生成的摘要和关键词单独保存在片段元数据中，检索结果保留原文。嵌入输入由原文、摘要及关键词组合而成。未变化的文件会跳过，避免重复调用。

普通文件共享不会触发自动入库。添加资料后，手动执行下列命令。未创建定时任务或后台扫描服务。

## 处理结果保存位置

| 内容 | 位置 |
|---|---|
| 原始 PDF、笔记、代码 | 保持在上面的来源目录 |
| 正文片段、页码、摘要、关键词、向量和索引 | 本机 PostgreSQL 数据库 `vector_library`，物理数据目录 `F:\vector_library\pgdata` |
| 当前云端管道的逻辑表 | `public.knowledge_qwen37_enriched_documents`、`public.knowledge_qwen37_enriched_chunks`、`public.knowledge_qwen37_enriched_settings` |
| 数据库备份 | `F:\vector_library\backups\*.dump` |
| 首篇论文的检索验证报告 | `F:\vector_library\logs\paper-ingestion-20261025.json` |

日常通过 `search`、`documents`、`stats` 命令或 PostgreSQL 客户端读取结果。`pgdata` 由 PostgreSQL 管理，不需要手动打开或编辑其中的文件。检索结果默认输出到终端；不会在来源目录自动生成另一份向量文件。多个来源共用同一知识库，也共用该库的访问权限。

## 完整云端模型入库与检索（当前推荐入口）

PowerShell 中执行：

```powershell
cd E:\vector_library\vector_library
& .\scripts\kvlib-cloud.ps1 ingest
& .\scripts\kvlib-cloud.ps1 search "这里输入要检索的问题"
& .\scripts\kvlib-cloud.ps1 stats
& .\scripts\kvlib-cloud.ps1 documents
```

不带来源路径的 `ingest` 会一次处理本机资料、共享资料与共享代码；以上 `search` 同时检索这三处已经入库的内容。

指定一个文件或其他目录：

```powershell
& .\scripts\kvlib-cloud.ps1 ingest 'F:\vector_library\shared\code'
& .\scripts\kvlib-cloud.ps1 search '数据库事务' --mode lexical
```

## 完全本地模式

```powershell
& .\scripts\kvlib.ps1 ingest
& .\scripts\kvlib.ps1 search "这里输入要检索的问题"
& .\scripts\kvlib.ps1 stats
```

各脚本访问各自的表，入库和搜索应使用相同脚本。`kvlib-cloud.ps1` 不依赖 Ollama。原有 `kvlib-qwen.ps1` 仍用于云端摘要加本地向量；它和完全本地模式都需要 Ollama 运行且已下载 `bge-m3:latest`。

当前 Python 为 `F:\vector_library\pipeline-venv\Scripts\python.exe`，项目以 editable 方式安装，因此修改 E 盘源码会直接生效。数据库服务继续使用现有自动启动配置。

## 配置和密钥

- 本地管道：`.knowledge-vector/config.toml`
- Qwen 增强：`.knowledge-vector/qwen.toml`
- 完整云端模型：`.knowledge-vector/cloud.toml`
- 仅云端向量：`.knowledge-vector/cloud-embedding.toml`
- 数据库密码沿用 `F:\vector_library\secrets\credentials.json` 的 Windows 账号。
- Qwen 密钥已从用户提供的文件读取并存入 `F:\vector_library\secrets\qwen-token-plan.key`，也可用 `KVLIB_QWEN_API_KEY` 环境变量覆盖。
- 新向量模型密钥已从 `E:\vector_library\qwen-vector.txt` 读取并存入 `F:\vector_library\secrets\bailian-embedding.key`，也可用 `KVLIB_EMBEDDING_API_KEY` 覆盖。两个模型使用各自的密钥与接口。
- 本地配置及凭据不加入 Git。原始 `api_qwen.txt` 和 `qwen-vector.txt` 保持在原位置，已按文件名排除入库。

原文、片段、页码、标题、模型和处理参数保存在 PostgreSQL 中；每篇文档替换在一个事务中完成。单篇失败不会清空旧版本，CLI 返回非零退出码并列出失败文件。文件正文、侧车元数据、切片参数、模型或 Qwen 提示词版本改变时会重新处理。当前不自动删除从源目录移走的文档。

云端请求遇到连接中断、响应截断或网络超时会最多尝试3次；HTTP状态错误会直接报错。本次论文导入的进度与中间标注结果保存在主机 `F:\vector_library\logs`，不在 SMB 共享目录中。

使用 HNSW 余弦向量索引与 GIN 全文索引。中文关键词使用字词 n-gram，不等同于专业中文分词器。每个表前缀固定一个模型和维度，切换模型需要新前缀并重新导入。

## 云端向量模型入口

2026-10-25 已成功调用新文件指定的 `qwen3.7-text-embedding-flash`，通过工作空间专属域名的 OpenAI 兼容 `/embeddings` 接口返回 1024 维向量。请求显式发送 `dimensions = 1024`、`encoding_format = "float"`，每批最多8条输入。

新接口已写入本机配置。公开模板 `config.cloud.example.toml`（含摘要）与 `config.cloud-embedding.example.toml`（仅向量）使用工作空间占位符，可供其他机器部署。

如需减少摘要调用，只使用云端向量，可执行：

```powershell
& F:\vector_library\pipeline-venv\Scripts\python.exe -m knowledge_vector init --config .knowledge-vector/cloud-embedding.toml
& F:\vector_library\pipeline-venv\Scripts\python.exe -m knowledge_vector ingest --config .knowledge-vector/cloud-embedding.toml
& F:\vector_library\pipeline-venv\Scripts\python.exe -m knowledge_vector search "检索问题" --config .knowledge-vector/cloud-embedding.toml
```

云端嵌入会发送片段正文及检索问题至配置的云端接口；完整管道还会把原文发送给摘要模型。文件解析和数据库存储仍在本机执行。两种管道使用独立表前缀，旧 BGE-M3 向量不会混入新模型的数据表。

参考：[Qwen3.7向量模型说明](https://help.aliyun.com/zh/model-studio/qwen3-7-text-embedding-flash)、[兼容接口](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)。

## 验证与备份

前次本地管道21项测试通过。本次云端配置运行22项测试全部通过，包括真实云端向量多格式入库、向量/关键词/混合检索、重复导入跳过、内容更新、事务失败回滚、模型维度不匹配拒绝，以及实际 qwen3.8-flash → qwen3.7-text-embedding-flash → pgvector 全链路。测试使用临时文件和独立随机表，结束后删除，验证完成时测试数据已清理；正式库内容以 stats 命令输出为准。

测试报告：`F:\vector_library\logs\pipeline-tests.xml`（前次本地）、`F:\vector_library\logs\cloud-pipeline-tests.xml`（本次云端）。实际网络测试需要显式设置 `KVLIB_TEST_CONFIG`（基础入库配置）和 `KVLIB_QWEN_TEST_CONFIG`（含摘要的配置），默认测试不会调用云端。

原有 `F:\vector_library\scripts\backup.ps1` 会备份新建的数据表。原始文件及共享目录仍需单独备份。

多来源更新验证：基础测试与真实云端多来源入库共28项通过，4项其他可选网络测试跳过。验证覆盖路径去重、重复导入跳过、指定单目录、单文件失败不阻断其他来源，以及本机资料、共享资料与共享代码写入同一 PostgreSQL 表集。报告：`F:\vector_library\logs\multi-source-tests.xml`。

## 同步到 GitHub

仓库：`https://github.com/kungaocode/vector_library`，分支：`main`。Git 同步项目源码、测试、配置模板和操作文档；本机配置、API Key、数据库密码、原始资料、数据库数据和备份单独保留在主机上。

后续修改后的同步命令：

```powershell
cd E:\vector_library\vector_library
git -c safe.directory=E:/vector_library/vector_library status
git -c safe.directory=E:/vector_library/vector_library add .
git -c safe.directory=E:/vector_library/vector_library commit -m "Describe the update"
git -c safe.directory=E:/vector_library/vector_library push origin main
```

本项目位于 exFAT 的 E 盘，上面命令中的 `safe.directory` 只对当前 Git 命令生效。其他电脑克隆仓库后，需另外创建本机配置并设置密钥、数据库和原始资料路径；GitHub 代码同步不等同于数据库备份。

本机 Git 已在该仓库的本地配置中使用 Windows 代理地址 `http://127.0.0.1:9674`，代理客户端需要运行。若代理端口变化，可用 `git -c safe.directory=E:/vector_library/vector_library config --local http.proxy 新地址` 更新；不再使用代理时，可用同一命令的 `--unset http.proxy` 删除该项。此配置不上传到 GitHub。
