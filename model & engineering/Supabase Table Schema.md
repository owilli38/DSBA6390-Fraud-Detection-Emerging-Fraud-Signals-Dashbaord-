## Table `alert_rules`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `rule_name` | `text` |  |
| `rule_type` | `text` |  |
| `threshold` | `numeric` |  Nullable |
| `description` | `text` |  Nullable |
| `is_active` | `bool` |  Nullable |
| `created_at` | `timestamptz` |  Nullable |

## Table `alerts`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `cluster_id` | `int4` |  Nullable |
| `rule_id` | `uuid` |  Nullable |
| `severity` | `text` |  |
| `alert_type` | `text` |  |
| `message` | `text` |  Nullable |
| `cluster_risk_snapshot` | `numeric` |  Nullable |
| `cluster_stage_snapshot` | `text` |  Nullable |
| `triggered_at` | `timestamptz` |  Nullable |
| `is_read` | `bool` |  Nullable |

## Table `article_analysis`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `doc_id` | `text` | Primary |
| `bert_embeddings` | `vector` |  Nullable |
| `cluster_id` | `int4` |  Nullable |
| `age_days` | `numeric` |  Nullable |
| `growth` | `numeric` |  Nullable |
| `drift` | `numeric` |  Nullable |
| `acceleration` | `numeric` |  Nullable |
| `stage` | `text` |  Nullable |
| `risk_score` | `numeric` |  Nullable |
| `nearest_article` | `bool` |  Nullable |

## Table `article_embeddings`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `doc_id` | `text` | Primary |
| `embedding` | `vector` |  Nullable |

## Table `article_embeddings_v2`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `doc_id` | `text` | Primary |
| `embedding` | `vector` |  Nullable |

## Table `article_neighbors`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `doc_id` | `text` | Primary |
| `neighbor_doc_id` | `text` | Primary |
| `similarity_score` | `numeric` |  Nullable |
| `rank` | `int4` |  Nullable |

## Table `articles_v1`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `doc_id` | `text` | Primary |
| `source` | `text` |  Nullable |
| `source_type` | `text` |  Nullable |
| `publish_timestamp` | `text` |  Nullable |
| `title` | `text` |  Nullable |
| `raw_text` | `text` |  Nullable |
| `url` | `text` |  Nullable Unique |
| `fetch_timestamp` | `text` |  Nullable |
| `raw_html` | `text` |  Nullable |

## Table `cluster_themes`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `cluster_id` | `int4` | Primary |
| `theme_label` | `text` |  Nullable |
| `theme_description` | `text` |  Nullable |
| `theme_evidence` | `jsonb` |  Nullable |
| `article_count` | `int4` |  Nullable |
| `stage` | `text` |  Nullable |
| `risk_score` | `numeric` |  Nullable |
| `created_at` | `timestamptz` |  Nullable |
| `updated_at` | `timestamptz` |  Nullable |

## Table `raw_articles`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `doc_id` | `text` | Primary |
| `source` | `text` |  Nullable |
| `source_type` | `text` |  Nullable |
| `fetch_timestamp` | `timestamptz` |  Nullable |
| `publish_timestamp` | `text` |  Nullable |
| `title` | `text` |  Nullable |
| `raw_text` | `text` |  Nullable |
| `raw_html` | `text` |  Nullable |
| `url` | `text` |  Nullable Unique |

## Table `watchlists`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `cluster_id` | `int4` |  Nullable |
| `label` | `text` |  Nullable |
| `notes` | `text` |  Nullable |
| `created_at` | `timestamptz` |  Nullable |

