# DataMorgana数据生成器

基于论文《Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana》的中文复现实现。

## 功能特点

- **高度可配置**: 通过JSON配置文件定义用户和问题分类
- **多样化生成**: 基于多种分类的组合生成多样化的Q&A对
- **线程池加速**: 支持并发处理，提高生成效率
- **质量过滤**: 内置过滤机制确保生成质量
- **中文友好**: 支持中文文档和问题生成

## 系统架构

DataMorgana采用两阶段生成流程：

1. **配置阶段**: 定义用户分类和问题分类
2. **生成阶段**: 
   - 按概率选择类别组合
   - 随机选择文档
   - 调用LLM生成候选Q&A对
   - 过滤和选择最佳结果

## 安装要求

```bash
pip install openai loguru tqdm
```

## 文件结构

```
├── datamorgana_generator.py          # 主生成器文件
├── datamorgana_config_template.json  # 通用配置模板
├── healthcare_config.json            # 医疗领域配置示例
├── documents_sample.json             # 示例文档数据
└── README.md                         # 使用说明
```

## 快速开始

### 1. 基本使用

```python
from datamorgana_generator import DataMorganaGenerator, Config

# 创建配置
config = Config()
config.input_path = "documents_sample.json"
config.output_path = "generated_qa.json"
config.config_path = "datamorgana_config_template.json"

# 创建生成器
generator = DataMorganaGenerator(config)

# 加载配置
generator.load_configuration(config.config_path)

# 加载文档并生成
import json
with open(config.input_path, 'r', encoding='utf-8') as f:
    documents = json.load(f)

results = generator.generate_benchmark(documents)

# 保存结果
with open(config.output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 2. 命令行使用

```bash
python datamorgana_generator.py
```

## 配置文件说明

### 基本结构

```json
{
  "user_categorizations": [
    {
      "name": "分类名称",
      "categories": [
        {
          "name": "类别名称",
          "probability": 0.5,
          "description": "类别描述"
        }
      ]
    }
  ],
  "question_categorizations": [
    // 问题分类配置...
  ]
}
```

### 默认问题分类（基于论文Table 1）

1. **Factuality（事实性）**
   - `factoid`: 寻求具体事实信息的问题
   - `open-ended`: 开放性讨论问题

2. **Premise（前提）**
   - `direct`: 直接问题，不包含用户信息
   - `with-premise`: 包含用户背景信息的问题

3. **Phrasing（措辞风格）**
   - `concise-and-natural`: 简洁自然语言（<10词）
   - `verbose-and-natural`: 详细自然语言（>9词）
   - `short-search-query`: 短搜索查询（<7词）
   - `long-search-query`: 长搜索查询（>6词）

4. **Linguistic Variation（语言变化）**
   - `similar-to-document`: 使用文档相似术语
   - `distance-from-document`: 使用不同术语

5. **Language（语言选择）** ⭐
   - `chinese`: 生成中文问题和答案（概率0.7）
   - `english`: 生成英文问题和答案（概率0.3）

### 用户分类示例

**通用分类**:
- `expert`: 专业用户
- `novice`: 普通用户

**医疗领域分类**:
- `patient`: 患者
- `medical-doctor`: 医生
- `clinical-researcher`: 临床研究员
- `public-health-authority`: 公共卫生部门

## 自定义配置

### 1. 创建领域特定配置

为教育领域创建配置：

```json
{
  "user_categorizations": [
    {
      "name": "education_role",
      "categories": [
        {
          "name": "student",
          "probability": 0.6,
          "description": "a student seeking to learn new concepts"
        },
        {
          "name": "teacher",
          "probability": 0.3,
          "description": "a teacher looking for educational content"
        },
        {
          "name": "researcher",
          "probability": 0.1,
          "description": "an academic researcher studying the field"
        }
      ]
    }
  ]
}
```

### 2. 调整概率分布

根据预期用户流量调整类别概率：

```json
{
  "name": "factuality",
  "categories": [
    {
      "name": "factoid",
      "probability": 0.3,  // 降低事实性问题比例
      "description": "..."
    },
    {
      "name": "open-ended", 
      "probability": 0.7,  // 增加开放性问题比例
      "description": "..."
    }
  ]
}
```

## 参数配置

### Config类参数

```python
@dataclass
class Config:
    api_key: str = "your-api-key"              # API密钥
    base_url: str = "https://openai.com/v1"  # API基础URL
    model: str = "claude-3-7-sonnet-20250219"  # 使用的模型
    input_path: str = "documents.json"         # 输入文档路径
    output_path: str = "output.json"           # 输出结果路径
    config_path: str = "config.json"           # 配置文件路径
    max_retries: int = 3                       # 最大重试次数
    retry_delay: int = 2                       # 重试延迟(秒)
    api_call_delay: float = 1                  # API调用间隔(秒)
    max_workers: int = 4                       # 线程池大小
    num_questions_per_document: int = 1        # 每文档生成问题数
    candidate_questions_per_call: int = 3      # 每次调用生成候选数
```

## 输出格式

生成的结果包含以下信息：

```json
[
  {
    "id": "文档ID",
    "content": "文档内容",
    "generated_qa_pairs": [
      {
        "question": "生成的问题",
        "answer": "生成的答案",
        "user_categories": "expert",
        "question_categories": "factoid, direct, concise-and-natural, similar-to-document",
        "document_id": "文档ID"
      }
    ]
  }
]
```

## 性能优化

### 1. 并发控制

```python
config.max_workers = 8  # 增加并发数
config.api_call_delay = 0.5  # 减少API调用间隔
```

### 2. 批量生成

```python
config.num_questions_per_document = 5  # 每文档生成更多问题
config.candidate_questions_per_call = 5  # 增加候选数
```

### 3. 错误处理

```python
config.max_retries = 5  # 增加重试次数
config.retry_delay = 3  # 增加重试延迟
```

## 质量控制

系统内置多层质量过滤：

1. **基本质量检查**: 问题和答案长度验证
2. **上下文无关性**: 确保问题不引用文档本身
3. **问题格式验证**: 检查是否为合法问题格式
4. **多样性保证**: 通过分类组合确保多样性

## 对比实验

与其他方法的多样性对比（基于论文指标）：

- **N-Gram Diversity (NDG)**: 词汇多样性
- **Self-Repetition Score (SRS)**: 重复率
- **Compression Ratio (CR)**: 压缩率
- **Homogenization Score (HS)**: 语义相似度

## 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥和网络连接
   - 适当增加重试次数和延迟

2. **解析错误**
   - 检查模型输出格式
   - 调整prompt模板

3. **生成质量差**
   - 调整过滤条件
   - 优化分类描述

### 日志配置

```python
from loguru import logger
logger.add("datamorgana.log", rotation="1 MB")
```

## 扩展功能

### 1. 自定义过滤器

```python
def custom_filter(qa_pairs, document):
    # 实现自定义过滤逻辑
    return filtered_pairs

generator.filter_qa_pairs = custom_filter
```

### 2. 自定义prompt模板

```python
def custom_prompt_builder(document, user_cats, question_cats, num_q):
    # 实现自定义prompt构建
    return prompt

generator.build_prompt = custom_prompt_builder
```

## 引用

如果使用此实现，请引用原论文：

```
Filice, S., Horowitz, G., Carmel, D., Karnin, Z., Lewin-Eytan, L., & Maarek, Y. 
Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana.
```

## 许可证

本项目基于Apache 2.0许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进此实现。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。 