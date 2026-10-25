# Knowledge Vector Library

一个本地优先的个人知识向量库基础实现。它负责把 PDF、论文、Markdown、文本和百科 HTML 转换为可追溯的知识片段，并通过关键词与向量混合检索找回相关内容。

详细设计见 [PROJECT.md](PROJECT.md)。

## 当前能力

- 增量导入：内容未变化时跳过，变化时替换旧片段。
- 多格式读取：`.pdf`、`.html`、`.htm`、`.md`、`.markdown`、`.txt`、`.json`、`.jsonl`。
- SQLite 存储：文档、片段、元数据、FTS5 全文索引和向量。
- 混合检索：关键词分数与向量分数归一化后融合。
- 可替换嵌入器：默认使用无外部依赖的哈希嵌入器，后续可切换语义模型。

## 快速开始

```bash
git clone git@github.com:kungaocode/vector_library.git
cd vector_library
python -m pip install -e ".[dev]"

kvlib init
kvlib ingest sources/
kvlib search "注意力机制"
kvlib stats
```

若要自定义切片、嵌入或检索参数，可在初始化后参考 `config.example.toml` 修改
`.knowledge-vector/config.toml`。

也可以不安装，直接使用源码：

```bash
PYTHONPATH=src python -m knowledge_vector init
PYTHONPATH=src python -m knowledge_vector ingest sources/
PYTHONPATH=src python -m knowledge_vector search "向量检索"
```

## 源文件组织

推荐先按类型分目录，不必强制：

```text
sources/
├── papers/
├── books/
├── wiki/
└── notes/
```

HTML 或 Markdown 文件可以选择同名侧车元数据：

```text
attention-is-all-you-need.pdf
attention-is-all-you-need.pdf.meta.json
```

侧车文件示例：

```json
{
  "title": "Attention Is All You Need",
  "source_type": "paper",
  "authors": ["Vaswani et al."],
  "year": 2017,
  "url": "https://arxiv.org/abs/1706.03762",
  "tags": ["transformer", "attention"]
}
```

侧车中的 `title` 和 `source_type` 会覆盖自动推断值，其余字段进入元数据。

## CLI

```bash
# 初始化数据库目录
kvlib init

# 递归导入
kvlib ingest sources/ --recursive

# 将整个目录标记为论文
kvlib ingest sources/papers/ --source-type paper

# 混合检索
kvlib search "Transformer 中的注意力机制" --top-k 10

# 只使用向量或关键词
kvlib search "QKV" --mode vector
kvlib search "1706.03762" --mode lexical

# 查看数据库状态和已导入文档
kvlib stats
kvlib documents
```

## 嵌入模型

默认配置：

```toml
[embedding]
provider = "hashing"
model = "hashing-v1"
dimension = 384
api_url = ""
api_key = ""
timeout_s = 60
batch_size = 32
```

哈希嵌入器不需要下载模型，适合先验证管道，不是生产级语义检索方案。后续可安装可选依赖并切换：

```bash
python -m pip install -e ".[semantic]"
```

```toml
[embedding]
provider = "sentence-transformers"
model = "BAAI/bge-m3"
dimension = 1024
```

也可以保留一个 OpenAI-compatible 嵌入服务入口。`api_url` 应填写完整的 embeddings
endpoint；本地服务不要求鉴权时 `api_key` 可以留空：

```toml
[embedding]
provider = "openai-compatible"
model = "your-embedding-model"
dimension = 1024
api_url = ""
api_key = ""
timeout_s = 60
batch_size = 32
```

使用 `openai-compatible` 时，若 `api_url` 仍为空，初始化该 provider 会直接报告配置缺失。
默认的 `hashing` 模式不会访问网络；只做关键词检索时也不会初始化 embedding provider。

更换嵌入模型后，需要使用新数据库或重新构建向量索引，不能混用不同模型的向量。

## 测试

```bash
PYTHONPATH=src python -m pytest
```

## 下一阶段

真实 PDF、论文和百科入口放好以后，优先用它们检查：

1. 双栏论文的正文顺序。
2. 页眉、页脚、参考文献和公式噪声。
3. 中文技术语料的切片长度。
4. 语义嵌入模型的实际召回质量。
5. 围绕个人学习问题建立的小型检索评测集。
