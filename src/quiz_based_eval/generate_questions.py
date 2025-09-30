from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json
import os
import re
import json5
import random
import numpy as np
import yaml
from pathlib import Path


def drop_ref(text):
    """
    移除文本中最后出现的 References 部分
    支持多种格式：
    - ## References / ## REFERENCES (Markdown 二级标题)
    - # References / # REFERENCES (Markdown 一级标题)
    - ### References (更多#号)
    - References: / REFERENCES: (带冒号)
    - **References** (粗体格式)
    - 大小写混合 (References, references, REFERENCES)
    """
    # 定义多种可能的 References 标记模式
    # 使用正则表达式匹配各种格式，(?i) 表示不区分大小写
    patterns = [
        r'#{1,6}\s*references?\s*$',  # Markdown 标题格式 (1-6个#)
        r'\*\*references?\*\*\s*$',  # 粗体格式
        r'references?\s*:\s*$',  # 带冒号格式
        r'references?\s*$',  # 纯文本格式
    ]

    # 查找所有可能的 References 标记位置
    matches = []
    for pattern in patterns:
        for match in re.finditer(f'(?im)^.*{pattern}', text):
            matches.append(match.start())

    # 如果找到匹配项，取最后一个位置
    if matches:
        last_ref_pos = max(matches)
        return text[:last_ref_pos].strip()

    return text.strip()


@dataclass
class HeadingInfo:
    """标题信息数据类"""
    level: int
    title: str
    start_pos: int
    end_pos: Optional[int] = None


class MarkdownParser:
    """Markdown文档解析器，用于提取标题、段落并构建索引"""
    def __init__(self):
        self.entries = []
        self.max_sentence_length = 200
        self.min_paragraph_length = 10
        self.supported_languages = ['en']
        self.sentence_patterns = r'(?<=[.!?])\s+'
        self.logger = logging.getLogger(__name__)

    def extract_headings(self, markdown_text: str) -> List[HeadingInfo]:
        """
                从Markdown文本中提取所有标题

                Args:
                    markdown_text: 输入的Markdown文本

                Returns:
                    按位置排序的标题信息列表
                """
        headings = []

        atx_pattern = re.compile(r'^(#{1,6})\s*(.*?)(?:\s*#+)?\s*$', re.MULTILINE)
        for match in atx_pattern.finditer(markdown_text):
            level = len(match.group(1))
            title = match.group(2).strip()
            if title:
                headings.append(HeadingInfo(
                    level=level,
                    title=title,
                    start_pos=match.start()
                ))

        setext_pattern = re.compile(r'^(.+)\n(=+|-+)\s*$', re.MULTILINE)
        for match in setext_pattern.finditer(markdown_text):
            title = match.group(1).strip()
            underline = match.group(2)
            level = 1 if underline[0] == '=' else 2
            if title:
                headings.append(HeadingInfo(
                    level=level,
                    title=title,
                    start_pos=match.start()
                ))

        headings.sort(key=lambda x: x.start_pos)

        for i, heading in enumerate(headings):
            if i + 1 < len(headings):
                heading.end_pos = headings[i + 1].start_pos
            else:
                heading.end_pos = len(markdown_text)

        return headings

    def build_heading_hierarchy(self, headings: List[HeadingInfo]) -> List[Dict[str, Any]]:
        """
        构建标题的层级关系树

        Args:
            headings: 标题信息列表

        Returns:
            包含层级路径信息的标题字典列表
        """
        hierarchy = []
        stack = []

        for heading in headings:
            while stack and stack[-1]['level'] >= heading.level:
                stack.pop()

            path = [item['title'] for item in stack] + [heading.title]
            parent_title = stack[-1]['title'] if stack else None

            heading_info = {
                'level': heading.level,
                'title': heading.title,
                'parent_title': parent_title,
                'path': path,
                'full_path': ' > '.join(path),
                'start_pos': heading.start_pos,
                'end_pos': heading.end_pos
            }

            hierarchy.append(heading_info)
            stack.append(heading_info)

        return hierarchy

    def extract_main_sentence(self, text: str) -> str:
        """
        智能提取段落的中心句

        Args:
            text: 输入的段落文本

        Returns:
            提取的中心句
        """
        if not text or len(text) < self.min_paragraph_length:
            return text.strip()

        clean_text = self.clean_markdown(text)
        sentences = self._smart_sentence_split(clean_text)

        if not sentences:
            return clean_text[:self.max_sentence_length]

        first_sentence = sentences[0]
        if len(first_sentence) > 20 and self._contains_key_indicators(first_sentence):
            return first_sentence

        if len(sentences) > 1:
            for s in sentences[1]:
                if len(s) > len(first_sentence) * 1.5 and self._contains_key_indicators(s):
                    return s

        longest_sentence = max(sentences, key=len)
        if len(longest_sentence) > len(first_sentence) * 1.5:
            return longest_sentence

        return first_sentence

    def _smart_sentence_split(self, text: str) -> List[str]:
        """
        增强的句子分割方法

        Args:
            text: 输入文本

        Returns:
            分割后的句子列表
        """
        text = self._preprocess_text(text)
        sentences = []

        primary_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff])', text)

        for segment in primary_splits:
            segment = segment.strip()
            if not segment:
                continue

            secondary_splits = self._handle_secondary_delimiters(segment)
            sentences.extend(secondary_splits)

        return self._postprocess_sentences(sentences)

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本，保护常见缩写和特殊格式

        Args:
            text: 原始文本

        Returns:
            预处理后的文本
        """
        # 保护常见的缩写（如 Mr. Dr. etc.）
        abbreviations = [
            r'\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|Inc|Ltd|Co|Corp|etc|vs|e\.g|i\.e|cf|al|St|Ave|Rd|Blvd)\.',
            r'\b[A-Z]\.(?:[A-Z]\.)+',  # 如 U.S.A., N.Y.C.
            r'\b\d+\.\d+',  # 数字如 3.14, 2.5
            r'(?<=\d)\.\s*(?=\d)',  # 数字中间的点
        ]

        protected = {}
        for i, pattern in enumerate(abbreviations):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                placeholder = f"__PROTECTED_{i}_{len(protected)}__"
                protected[placeholder] = match.group()
                text = text.replace(match.group(), placeholder, 1)

        self._protected_content = protected
        return text

    def _handle_secondary_delimiters(self, segment: str) -> List[str]:
        """
        处理次级分隔符（冒号、分号）

        Args:
            segment: 文本片段

        Returns:
            分割后的句子列表
        """
        results = []

        if ':' in segment:
            parts = segment.split(':')
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                if i == 0 and len(part) < 15 and i + 1 < len(parts):
                    combined = f"{part}: {parts[i + 1].strip()}"
                    if self._is_valid_sentence(combined):
                        results.append(combined)
                        parts[i + 1] = ""
                elif self._is_valid_sentence(part):
                    results.append(part)

        elif ';' in segment:
            parts = re.split(r';\s*', segment)
            for part in parts:
                part = part.strip()
                if self._is_valid_sentence(part):
                    results.append(part)

        else:
            if self._is_valid_sentence(segment):
                results.append(segment)

        return results if results else [segment]

    def _is_valid_sentence(self, text: str) -> bool:
        """
        判断文本是否为有效句子

        Args:
            text: 待判断文本

        Returns:
            是否为有效句子
        """
        if not text or len(text.strip()) < 3:
            return False

        text = text.strip()

        if re.match(r'^\d+\.?\s*$', text):
            return False

        if len(text.split()) == 1 and not re.search(r'[.!?]$', text):
            return False

        if not re.search(r'[a-zA-Z\u4e00-\u9fff]', text):
            return False

        return True

    def _postprocess_sentences(self, sentences: List[str]) -> List[str]:
        """
        后处理句子列表，恢复保护内容并清理格式

        Args:
            sentences: 句子列表

        Returns:
            处理后的句子列表
        """
        if not hasattr(self, '_protected_content'):
            self._protected_content = {}

        processed = []
        for sentence in sentences:
            if not sentence.strip():
                continue

            for placeholder, original in self._protected_content.items():
                sentence = sentence.replace(placeholder, original)

            sentence = re.sub(r'\s+', ' ', sentence.strip())

            if sentence and not re.search(r'[.!?:;]$', sentence):
                if len(sentence) > 10 and ' ' in sentence:
                    sentence += '.'

            if sentence:
                processed.append(sentence)

        if hasattr(self, '_protected_content'):
            delattr(self, '_protected_content')

        return processed

    def _contains_key_indicators(self, sentence: str) -> bool:
        """
        检查句子是否包含关键指示词

        Args:
            sentence: 待检查的句子

        Returns:
            是否包含关键指示词
        """
        key_indicators = {
            'en': ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'will', 'would',
                   'can', 'could', 'should', 'must', 'may', 'might', 'for', 'because',
                   'therefore', 'however', 'moreover', 'furthermore', 'in addition',
                   'represent', 'represents', 'represented', 'use', 'used',
                   'we', 'provide', 'provides', 'provided', 'effort', 'review', 'first']
        }

        sentence_lower = sentence.lower()

        for keyword in key_indicators['en']:
            if f' {keyword} ' in f' {sentence_lower} ':
                return True

        return False

    def clean_markdown(self, text: str) -> str:
        """
        清理Markdown格式标记

        Args:
            text: 包含Markdown标记的文本

        Returns:
            清理后的纯文本
        """
        # 移除标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # 移除粗体、斜体标记
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
        # 移除代码块
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        # 移除链接
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        # 移除图片
        text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', text)
        # 移除多余空白
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def split_into_paragraphs(self, text: str, heading_info: Dict[str, Any]) -> List[str]:
        """
        将文本按自然段落分割

        Args:
            text: 输入文本
            heading_info: 标题信息字典

        Returns:
            段落列表
        """
        content = text[text.find('\n'):] if text.find('\n') != -1 else text
        content = content.strip()

        if not content:
            return []

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        paragraphs = [p for p in paragraphs if len(p) >= self.min_paragraph_length]

        return paragraphs

    def validate_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证和过滤条目

        Args:
            entries: 待验证的条目列表

        Returns:
            验证通过的条目列表
        """
        valid_entries = []

        for entry in entries:
            try:
                required_fields = ['id', 'level', 'title', 'main_sentence', 'full_text']
                if not all(field in entry for field in required_fields):
                    self.logger.warning(f"条目 {entry.get('id', 'unknown')} 缺少必要字段")
                    continue

                if len(entry['full_text']) < self.min_paragraph_length:
                    self.logger.debug(f"条目 {entry['id']} 内容过短，跳过")
                    continue

                valid_entries.append(entry)

            except Exception as e:
                self.logger.error(f"验证条目时出错: {e}")
                continue

        return valid_entries

    def parse_and_build_index(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        优化的markdown解析和索引构建函数

        Args:
            markdown_text: 输入的markdown文本

        Returns:
            List[Dict[str, Any]]: 解析后的段落条目列表
        """
        try:
            if not markdown_text or not markdown_text.strip():
                self.logger.warning("输入的markdown文本为空")
                return []

            headings = self.extract_headings(markdown_text)
            if not headings:
                self.logger.warning("未找到任何标题，将整个文档作为一个段落处理")
                entries = [{
                    "id": 1,
                    "level": 0,
                    "title": "文档内容",
                    "parent_title": None,
                    "full_path": "文档内容",
                    "main_sentence": self.extract_main_sentence(markdown_text),
                    "full_text": markdown_text.strip(),
                    "paragraph_index": 1,
                    "paragraph_count": 1,
                    "word_count": len(markdown_text),
                    "metadata": {
                        "has_code": "```" in markdown_text,
                        "has_links": "[" in markdown_text and "](" in markdown_text,
                        "has_images": "![" in markdown_text
                    }
                }]
            else:
                hierarchy = self.build_heading_hierarchy(headings)

                entries = []
                entry_id = 1

                for heading_info in hierarchy:
                    start_pos = heading_info['start_pos']
                    end_pos = heading_info['end_pos']
                    section_text = markdown_text[start_pos:end_pos].strip()

                    paragraphs = self.split_into_paragraphs(section_text, heading_info)

                    if not paragraphs:
                        paragraphs = [section_text]

                    for i, paragraph in enumerate(paragraphs):
                        if not paragraph.strip():
                            continue

                        main_sentence = self.extract_main_sentence(paragraph)

                        entry = {
                            "id": entry_id,
                            "level": heading_info['level'],
                            "title": heading_info['title'],
                            "parent_title": heading_info['parent_title'],
                            "full_path": heading_info['full_path'],
                            "main_sentence": main_sentence,
                            "full_text": paragraph.strip(),
                            "paragraph_index": i + 1,
                            "paragraph_count": len(paragraphs),
                            "word_count": len(paragraph),
                            "metadata": {
                                "has_code": "```" in paragraph or "`" in paragraph,
                                "has_links": "[" in paragraph and "](" in paragraph,
                                "has_images": "![" in paragraph,
                                "section_start": start_pos,
                                "section_end": end_pos
                            }
                        }

                        entries.append(entry)
                        entry_id += 1

            entries = self.validate_entries(entries)

            if not entries:
                self.logger.error("没有有效的条目可以索引")
                return []

            self.entries = entries

            self.logger.info(f"成功解析 {len(entries)} 个段落条目")

            # main_sentences = [item["main_sentence"] for item in entries]
            # full_texts = [item["full_text"] for item in entries]
            # with open("../data/main_sentence.json", "w", encoding="utf-8") as f:
            #     json.dump(main_sentences, f, ensure_ascii=False, indent=4)
            #
            # with open("../data/full_text.json", "w", encoding="utf-8") as f:
            #     json.dump(full_texts, f, ensure_ascii=False, indent=4)

            return entries

        except Exception as e:
            self.logger.error(f"解析markdown文本时出错: {e}")
            raise


    def get_entry_by_id(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取条目

        Args:
            entry_id: 条目ID

        Returns:
            匹配的条目字典，未找到则返回None
        """
        for entry in self.entries:
            if entry['id'] == entry_id:
                return entry
        return None

    def get_entries_by_title(self, title: str) -> List[Dict[str, Any]]:
        """
        根据标题获取条目

        Args:
            title: 标题文本

        Returns:
            匹配的条目列表
        """
        return [entry for entry in self.entries if entry['title'] == title]

    def get_entries_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        根据标题层级获取条目

        Args:
            level: 标题层级（1-6）

        Returns:
            匹配的条目列表
        """
        return [entry for entry in self.entries if entry['level'] == level]

    def export_entries(self, format: str = 'json') -> str:
        """
        导出条目数据

        Args:
            format: 导出格式，支持'json'或'csv'

        Returns:
            导出的字符串数据
        """
        if format == 'json':
            import json
            return json.dumps(self.entries, ensure_ascii=False, indent=2)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            if self.entries:
                writer = csv.DictWriter(output, fieldnames=self.entries[0].keys())
                writer.writeheader()
                for entry in self.entries:
                    row = entry.copy()
                    row['metadata'] = str(row['metadata'])
                    writer.writerow(row)
            return output.getvalue()
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取解析统计信息

        Args:


        Returns:
            包含各种统计指标的字典
        """
        if not self.entries:
            return {}

        stats = {
            'total_entries': len(self.entries),
            'total_words': sum(entry['word_count'] for entry in self.entries),
            'level_distribution': {},
            'language_distribution': {},
            'avg_paragraph_length': np.mean([entry['word_count'] for entry in self.entries]),
            'entries_with_code': sum(1 for entry in self.entries if entry['metadata']['has_code']),
            'entries_with_links': sum(1 for entry in self.entries if entry['metadata']['has_links']),
            'entries_with_images': sum(1 for entry in self.entries if entry['metadata']['has_images'])
        }

        for entry in self.entries:
            level = entry['level']
            stats['level_distribution'][level] = stats['level_distribution'].get(level, 0) + 1

        for entry in self.entries:
            lang = entry['language']
            stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1

        return stats


class SurveyQAGenerator:
    """
    从survey中生成问答对的类
    使用基于规则的方法选取高质量段落
    使用LLM根据段落生成高质量的问题
    """

    def __init__(self, center_texts, full_texts, model, min_content_length: int = 100):
        """
        初始化问答生成器

        Args:
            center_texts: 存储段落中心句的索引
            full_texts: 存储段落全文的索引
            model: LLM模型实例，需要有send_message方法
            min_content_length: 最小内容长度，过短的内容会被过滤
        """
        self.center_texts = center_texts
        self.full_texts = full_texts
        self.model = model
        self.min_content_length = min_content_length

        if len(center_texts) != len(full_texts):
            raise ValueError("Center texts and full texts must have the same number of entries")

        self.total_segments = len(center_texts)

        self.logger = logging.getLogger(__name__)

    def _is_segment_suitable_for_qa(self, full_text: str, quality_threshold: float = 0.7) -> Tuple[
        bool, str, float]:
        """
        检查段落是否适合生成QA对，支持不同质量阈值

        设计思路： 分层判断 + 渐进式门槛
        对于当前检测指标：
        - 满足： 给满分，继续
        - 不满足但之前得分较多： 不得分，继续
        - 不满足且之前得分很少： 提前终止

        Args:
            full_text: 完整文本
            quality_threshold: 质量阈值 (0.0-1.0)

        Returns:
            (is_suitable, reason, quality_score)
        """
        quality_score = 0.0

        # 检查1：内容长度 (权重 0.1)
        if len(full_text.strip()) < self.min_content_length:
            if quality_threshold > 0.1:
                return False, "Content too short", 0.0
        else:
            quality_score += 0.1

        # 检查2：是否主要是公式 (权重 0.15)
        formula_patterns = [
            r'\$[^$]+\$',
            r'\\[a-zA-Z]+\{[^}]*\}',
            r'equation\s*\(\d+\)',
            r'[=<>≤≥≠±∑∏∫∂∇]+',
            r'\b[a-zA-Z]\s*[=<>]\s*[0-9a-zA-Z]+\b',
        ]

        text_without_formulas = full_text
        formula_count = 0
        for pattern in formula_patterns:
            matches = re.findall(pattern, full_text)
            formula_count += len(matches)
            text_without_formulas = re.sub(pattern, ' ', text_without_formulas)

        remaining_text = re.sub(r'\s+', ' ', text_without_formulas).strip()
        if formula_count <= 3 or len(remaining_text) >= 100:
            quality_score += 0.15
        elif quality_threshold > quality_score + 0.15:
            return False, "Too many formulas with insufficient explanatory text", quality_score

        # 检查3：是否主要是图片/表格引用 (权重 0.1)
        media_patterns = [
            r'\bfig(?:ure)?\s*\.?\s*\d+', r'\btable\s*\.?\s*\d+', r'\bimage\s*\.?\s*\d+',
            r'\bdiagram\s*\.?\s*\d+', r'\bgraph\s*\.?\s*\d+', r'see\s+fig(?:ure)?',
            r'shown\s+in\s+fig(?:ure)?', r'图\s*\d+', r'表\s*\d+', r'如图所示', r'见图',
        ]

        media_refs = sum(len(re.findall(pattern, full_text, re.IGNORECASE)) for pattern in media_patterns)
        if media_refs <= 2 or len(full_text.split()) >= 50:
            quality_score += 0.1
        elif quality_threshold > quality_score + 0.1:
            return False, "Too many media references with insufficient content", quality_score

        # 检查4：是否包含实质性内容 (权重 0.2)
        substantive_indicators = [
            r'\b(?:method|approach|technique|algorithm|process|procedure)\b',
            r'\b(?:result|finding|conclusion|observation|analysis)\b',
            r'\b(?:advantage|disadvantage|benefit|limitation|challenge)\b',
            r'\b(?:propose|demonstrate|show|prove|indicate|suggest)\b',
            r'\b(?:important|significant|critical|essential|key)\b',
            r'\b(?:because|since|therefore|thus|hence|consequently)\b',
            r'\b(?:however|although|while|whereas|nevertheless)\b'
        ]

        substantive_count = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE)) for pattern in substantive_indicators)
        if substantive_count >= 2:
            quality_score += 0.2
        elif substantive_count >= 1:
            quality_score += 0.1
        elif quality_threshold > quality_score + 0.2:
            return False, "Insufficient substantive content indicators", quality_score

        # 检查5：是否包含完整的句子 (权重 0.15)
        sentences = re.split(r'[.!?。！？]', full_text)
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 15 and ' ' in s.strip()]
        if len(complete_sentences) >= 2:
            quality_score += 0.15
        elif len(complete_sentences) >= 1:
            quality_score += 0.075
        elif quality_threshold > quality_score + 0.15:
            return False, "Insufficient complete sentences", quality_score

        # 检查6：避免纯列表或枚举 (权重 0.15)
        list_patterns = [
            r'^\s*[•·\-*]\s*',  # bullet points
            r'^\s*\d+[\.)]\s*',  # numbered lists
            r'^\s*[a-zA-Z][\.)]\s*',  # lettered lists
        ]

        lines = full_text.split('\n')
        list_lines = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line.strip()):
                    list_lines += 1
                    break

        if len(lines) == 0 or list_lines <= len(lines) * 0.6:
            quality_score += 0.15
        elif quality_threshold > quality_score + 0.15:
            return False, "Content is primarily a list", quality_score

        # 检查7：内容丰富度 (权重 0.15)
        word_count = len(full_text.split())
        unique_words = len(set(word.lower() for word in re.findall(r'\b\w+\b', full_text)))

        if word_count > 0:
            diversity_ratio = unique_words / word_count
            if diversity_ratio > 0.7:
                quality_score += 0.15
            elif diversity_ratio > 0.5:
                quality_score += 0.1
            elif diversity_ratio > 0.3:
                quality_score += 0.05

        is_suitable = quality_score >= quality_threshold
        reason = f"Quality score: {quality_score:.2f}, threshold: {quality_threshold:.2f}"

        return is_suitable, reason, quality_score

    def _get_random_segments_with_quality_check(self, n_segments: int, quality_threshold: float = 0.7,
                                                max_attempts: int = 100) -> List[Tuple[str, str, int]]:
        """
        随机选择n个高质量段落，自动过滤不适合生成QA的段落

        Args:
            n_segments: 需要的段落数量
            quality_threshold: 质量阈值
            max_attempts: 最大尝试次数

        Returns:
            List of (center_text, full_text, segment_id)
        """
        all_ids = np.arange(self.total_segments)
        candidate_ids = []

        for seg_id in all_ids:
            full_text = self.full_texts[seg_id]
            if len(full_text.strip()) >= max(50, self.min_content_length * 0.5):
                candidate_ids.append(seg_id)

        if len(candidate_ids) == 0:
            self.logger.warning("No candidate segments meet minimum length requirement")
            return []

        selected_segments = []
        used_ids = set()
        attempts = 0

        while len(selected_segments) < n_segments and attempts < max_attempts:
            attempts += 1

            available_ids = [seg_id for seg_id in candidate_ids if seg_id not in used_ids]
            if not available_ids:
                break

            seg_id = random.choice(available_ids)
            used_ids.add(seg_id)

            center_text = self.center_texts[seg_id]
            full_text = self.full_texts[seg_id]

            is_suitable, reason, quality_score = self._is_segment_suitable_for_qa(full_text,
                                                                                  quality_threshold)

            if is_suitable:
                selected_segments.append((center_text, full_text, seg_id))
                self.logger.debug(f"Selected segment {seg_id} with quality {quality_score:.2f}")
            else:
                self.logger.debug(f"Rejected segment {seg_id}: {reason}")

        return selected_segments

    def _get_any_available_segments(self, n_segments: int, used_segments: set,
                                    min_length: int = 50) -> List[Tuple[str, str, int]]:
        """
        获取任意可用段落（降低质量要求）
        """
        available_segments = []
        all_ids = list(range(self.total_segments))
        random.shuffle(all_ids)

        for seg_id in all_ids:
            if seg_id in used_segments or len(available_segments) >= n_segments:
                continue

            center_text = self.center_texts[seg_id]
            full_text = self.full_texts[seg_id]

            # 只检查基本长度
            if len(full_text.strip()) >= min_length:
                available_segments.append((center_text, full_text, seg_id))

        return available_segments

    def _generate_questions_for_segment(self, center_text: str, full_text: str,
                                        questions_per_segment: int = 1,
                                        avoid_duplicates: bool = False,
                                        existing_questions: List[str] = None,
                                        relaxed_mode: bool = False) -> List[Dict[str, str]]:
        """
        为单个段落生成问题（增强版，包含质量检查）
        """
        if existing_questions is None:
            existing_questions = []

        # 根据模式调整提示词
        if relaxed_mode:
            quality_instruction = "Generate practical questions that can be answered from the paragraph content."
            answer_requirement = "at least 1-2 sentences"
        else:
            quality_instruction = "Generate {questions_per_segment} high-quality questions."
            answer_requirement = "at least 2-3 sentences"

        # 去重指令
        duplicate_instruction = ""
        if avoid_duplicates and existing_questions:
            existing_q_sample = existing_questions[:3]
            duplicate_instruction = f"""
    IMPORTANT: Avoid generating questions similar to these existing ones:
    {chr(10).join(f'- {q}' for q in existing_q_sample)}
    """

        prompt = f"""Based on the following academic survey paragraph, {quality_instruction}

    Central sentence of the paragraph:
    {center_text}

    Full paragraph:
    {full_text}
    {duplicate_instruction}
    CRITICAL REQUIREMENTS:
    1. Each question MUST have a definitive, complete answer that can be fully derived from the given paragraph.
    2. Do NOT generate questions if the paragraph lacks sufficient context or information.
    3. Avoid questions about figures, tables, or equations that are referenced but not explained in the text.
    4. Questions should test understanding of concepts, methods, findings, or reasoning presented in the paragraph.
    5. Each answer must be substantial ({answer_requirement}) and directly based on the paragraph content.
    6. Questions MUST NOT contain ANY references to the source text, paragraph, paper, article, or any phrase that indicates where the information comes from.
    7. If you cannot generate {questions_per_segment} high-quality questions with complete answers, return fewer questions or an empty array.

    Question types to consider:
    - Concept definitions and explanations
    - Method descriptions and comparisons
    - Advantages/disadvantages analysis
    - Cause-and-effect relationships
    - Research findings and their implications

    Please strictly output in the following JSON format and include nothing else:

    [
      {{
        "question": "Specific, clear question that can be definitively answered",
        "answer": "Complete, detailed answer ({answer_requirement}) based entirely on the paragraph content"
      }},
      ...
    ]

    If the paragraph does not contain sufficient information to generate quality questions with complete answers, return an empty array: []"""

        try:
            message = [{"role": "user", "content": prompt}]
            response = self.model.send_message(message)

            if isinstance(response, str):
                response_text = response
            else:
                response_text = getattr(response, 'text', str(response))

            # 解析JSON
            try:
                match = re.search(r'\[[\s\S]*\]', response_text)
                if not match:
                    self.logger.warning("No JSON array found in response")
                    return []

                json_str = match.group(0)
                cleaned = re.sub(r'\$(.*?)\$', r'\1', json_str)
                if re.search(r'(?<!\\)\\(?![\\"])', cleaned):
                    cleaned = cleaned.replace('\\', '\\\\')

                qa_pairs = json5.loads(cleaned)

                print("Start evaluating qa_pair")
                # 验证QA质量
                valid_pairs = []
                if isinstance(qa_pairs, list):
                    for pair in qa_pairs:
                        if (isinstance(pair, dict) and 'question' in pair and 'answer' in pair):
                            question = str(pair['question']).strip()
                            answer = str(pair['answer']).strip()
                            # 去重检查
                            if avoid_duplicates and existing_questions:
                                is_duplicate = any(
                                    self._questions_are_similar(question, existing_q)
                                    for existing_q in existing_questions
                                )
                                if is_duplicate:
                                    continue

                            # 质量检查（降低标准）
                            if self._validate_qa_quality(question, answer, full_text, relaxed_mode=True):
                                valid_pairs.append({'question': question, 'answer': answer})

                return valid_pairs

            except json5.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}")
                return []

        except Exception as e:
            self.logger.error(f"Error generating questions for segment: {str(e)}")
            return []

    def _questions_are_similar(self, q1: str, q2: str, threshold: float = 0.7) -> bool:
        """检查两个问题是否相似"""
        # 简单的关键词相似度检查
        words1 = set(re.findall(r'\b\w+\b', q1.lower()))
        words2 = set(re.findall(r'\b\w+\b', q2.lower()))

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold

    def _validate_qa_quality(self, question: str, answer: str, source_text: str, relaxed_mode: bool = True) -> bool:
        """
        验证QA对的质量，支持宽松模式
        """
        # 基本长度要求（宽松模式下降低要求）
        min_q_len = 8 if relaxed_mode else 10
        min_a_len = 30 if relaxed_mode else 50

        if len(question.strip()) < min_q_len or len(answer.strip()) < min_a_len:
            return False

        # 检查问题和答案是否包含引用源文本的内容，确保问题自完整
        reference_patterns = [
            r'\bin\s+this\s+(?:paragraph|paper|text|study|article|document|section)\b',
            r'\baccording\s+to\s+(?:this|the)\s+(?:paragraph|paper|text|study|article|document|section)\b',
            r'\bthe\s+(?:paragraph|paper|text|study|article|document|section)\s+(?:states|mentions|discusses|describes|shows|indicates)\b',
            r'\bas\s+(?:mentioned|stated|described|shown|indicated)\s+in\s+(?:this|the)\s+(?:paragraph|paper|text|study|article|document|section)\b',
            r'\bfrom\s+(?:this|the)\s+(?:paragraph|paper|text|study|article|document|section)\b',
            r'\bthe\s+(?:above|following|given|provided)\s+(?:paragraph|text|passage)\b',
            r'\bthe\s+author[s]?\s+(?:state|mention|discuss|describe|show|indicate)\b',
            r'\bthe\s+research(?:ers)?\s+(?:found|discovered|showed|demonstrated)\b',
            r'\bthis\s+(?:paragraph|paper|text|study|article|document|section)\b',
            r'\bthe\s+(?:paragraph|paper|text|study|article|document|section)\b',
            r'\babove\s+(?:paragraph|text|passage|content)\b',
            r'\bmentioned\s+(?:paragraph|paper|text|study|article|document)\b',
        ]

        combined_text = f"{question} {answer}".lower()
        for pattern in reference_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                print(f"Reference Error: Found '{pattern}' in QA pair")
                return False

        # 答案实质内容检查（宽松模式下降低要求）
        answer_sentences = re.split(r'[.!?。！？]', answer)
        min_sentence_len = 15 if relaxed_mode else 20
        substantial_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > min_sentence_len]

        min_sentences = 1 if relaxed_mode else 2
        if len(substantial_sentences) < min_sentences:
            return False

        # 避免模糊问题
        vague_patterns = [
            r'\b(?:what|how|why)\s+(?:might|could|may|possibly)\b',
            r'\b(?:generally|usually|typically|often)\b',
            r'\bwhat\s+do\s+you\s+think\b',
            r'\bin\s+your\s+opinion\b',
        ]

        for pattern in vague_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                print("Vague Error")
                return False

        # 关键词重叠检查（宽松模式下降低要求）
        source_words = set(re.findall(r'\b\w+\b', source_text.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))

        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is',
                      'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        source_content_words = source_words - stop_words
        answer_content_words = answer_words - stop_words

        if len(source_content_words) > 0:
            overlap_ratio = len(source_content_words & answer_content_words) / len(source_content_words)
            min_overlap = 0.1 if relaxed_mode else 0.2
            if overlap_ratio < min_overlap:
                return False

        return True

    def generate_qa_pairs(self, n_questions: int, questions_per_segment: int = 1,
                          max_retries: int = 3, max_segment_attempts: int = 500) -> List[Dict[str, str]]:
        """
        生成N个高质量问答对，确保达到目标数量

        Args:
            n_questions: 需要生成的问题总数（必须达到此数量）
            questions_per_segment: 每个段落生成的问题数量
            max_retries: 每个段落的最大重试次数
            max_segment_attempts: 寻找合适段落的最大尝试次数

        Returns:
            List of {"question": str, "answer": str, "segment_id": int, "center_text": str}
        """
        self.logger.info(f"Starting to generate {n_questions} high-quality question-answer pairs")

        all_qa_pairs = []
        used_segments = set()
        total_attempts = 0
        quality_threshold = 0.7
        min_quality_threshold = 0.3
        relaxed_mode = False

        # 阶段1：正常质量生成
        while len(all_qa_pairs) < n_questions and total_attempts < max_segment_attempts:
            total_attempts += 1
            remaining_questions = n_questions - len(all_qa_pairs)

            if total_attempts > max_segment_attempts * 0.3 and not relaxed_mode:
                self.logger.info("Entering relaxed mode due to insufficient progress")
                relaxed_mode = True
                quality_threshold = max(quality_threshold - 0.2, min_quality_threshold)

            if total_attempts > max_segment_attempts * 0.6:
                quality_threshold = max(quality_threshold - 0.1, min_quality_threshold)

            needed_segments = min((remaining_questions + questions_per_segment - 1) // questions_per_segment, 10)
            segments = self._get_random_segments_with_quality_check(needed_segments, quality_threshold)

            if not segments and quality_threshold > min_quality_threshold:
                quality_threshold = max(quality_threshold - 0.1, min_quality_threshold)
                self.logger.info(f"Lowering quality threshold to {quality_threshold:.2f}")
                segments = self._get_random_segments_with_quality_check(needed_segments, quality_threshold)

            if not segments:
                self.logger.warning("Getting any available segments")
                segments = self._get_any_available_segments(needed_segments, used_segments)

            if not segments:
                self.logger.warning("No segments available, resetting used segments")
                if len(used_segments) > self.total_segments * 0.8:  # 如果使用了超过80%的段落
                    used_segments.clear()
                    continue
                else:
                    break

            segments_processed = 0
            for center_text, full_text, seg_id in segments:
                if seg_id in used_segments:
                    continue

                used_segments.add(seg_id)
                segments_processed += 1

                remaining_questions = n_questions - len(all_qa_pairs)
                if remaining_questions <= 0:
                    break

                current_questions_per_segment = min(questions_per_segment, remaining_questions)
                if remaining_questions <= 3:
                    current_questions_per_segment = remaining_questions

                self.logger.info(
                    f"Processing segment {seg_id} (attempt {total_attempts}, remaining: {remaining_questions})")

                success = False
                for retry in range(max_retries):
                    try:
                        qa_pairs = self._generate_questions_for_segment(
                            center_text, full_text, current_questions_per_segment,
                            avoid_duplicates=True,
                            existing_questions=[pair['question'] for pair in all_qa_pairs],
                            relaxed_mode=relaxed_mode
                        )

                        if qa_pairs and len(qa_pairs) > 0:
                            for pair in qa_pairs:
                                pair['segment_id'] = seg_id
                                pair['center_text'] = center_text

                            needed = n_questions - len(all_qa_pairs)
                            qa_pairs_to_add = qa_pairs[:needed]
                            all_qa_pairs.extend(qa_pairs_to_add)

                            success = True
                            self.logger.info(f"Generated {len(qa_pairs_to_add)} QA pairs from segment {seg_id}")
                            break
                        else:
                            self.logger.warning(f"No valid questions from segment {seg_id}, retry {retry + 1}")

                    except Exception as e:
                        self.logger.error(f"Error processing segment {seg_id}, retry {retry + 1}: {str(e)}")

                if not success:
                    self.logger.warning(
                        f"Failed to generate questions for segment {seg_id} after {max_retries} retries")

                if len(all_qa_pairs) >= n_questions:
                    break

            if segments_processed == 0:
                self.logger.warning("No new segments processed")
                # 重置使用过的段落以允许重复使用
                if len(all_qa_pairs) < n_questions * 0.7:
                    used_segments.clear()

        # 阶段2：兜底策略 - 从已成功的段落重新生成
        if len(all_qa_pairs) < n_questions and len(all_qa_pairs) > 0:
            self.logger.warning(f"Attempting fallback generation. Current: {len(all_qa_pairs)}/{n_questions}")
            remaining_needed = n_questions - len(all_qa_pairs)

            successful_segments = list(set(pair['segment_id'] for pair in all_qa_pairs))
            random.shuffle(successful_segments)

            for seg_id in successful_segments:
                if len(all_qa_pairs) >= n_questions:
                    break

                matching_pair = next(pair for pair in all_qa_pairs if pair['segment_id'] == seg_id)
                center_text = matching_pair['center_text']
                full_text = self.full_texts[seg_id]

                try:
                    additional_qa_pairs = self._generate_questions_for_segment(
                        center_text, full_text,
                        min(3, remaining_needed),
                        avoid_duplicates=True,
                        existing_questions=[pair['question'] for pair in all_qa_pairs],
                        relaxed_mode=True
                    )

                    if additional_qa_pairs:
                        for pair in additional_qa_pairs:
                            pair['segment_id'] = seg_id
                            pair['center_text'] = center_text

                        needed = n_questions - len(all_qa_pairs)
                        all_qa_pairs.extend(additional_qa_pairs[:needed])
                        self.logger.info(
                            f"Fallback generated {min(len(additional_qa_pairs), needed)} additional QA pairs")

                except Exception as e:
                    self.logger.error(f"Error in fallback generation: {str(e)}")

        # 阶段3：最后的兜底 - 降到最低标准
        if len(all_qa_pairs) < n_questions * 0.8:  # 如果连80%都达不到
            self.logger.warning("Attempting emergency generation with minimal standards")
            remaining = n_questions - len(all_qa_pairs)

            emergency_segments = self._get_any_available_segments(remaining * 2, set(), min_length=30)

            for center_text, full_text, seg_id in emergency_segments:
                if len(all_qa_pairs) >= n_questions:
                    break

                try:
                    qa_pairs = self._generate_questions_for_segment(
                        center_text, full_text, 1,
                        avoid_duplicates=False,
                        relaxed_mode=True
                    )

                    if qa_pairs:
                        for pair in qa_pairs:
                            pair['segment_id'] = seg_id
                            pair['center_text'] = center_text
                        all_qa_pairs.extend(qa_pairs[:1])  # 只取一个

                except Exception:
                    continue

        final_count = len(all_qa_pairs)
        self.logger.info(f"Final result: generated {final_count} question-answer pairs (target: {n_questions})")

        if final_count < n_questions:
            self.logger.error(f"FAILED to reach target: only generated {final_count}/{n_questions} QA pairs")
            completion_rate = (final_count / n_questions) * 100
            self.logger.error(f"Completion rate: {completion_rate:.1f}%")

        return all_qa_pairs[:n_questions]  # 确保不超过目标数量


    def save_qa_pairs(self, qa_pairs: List[Dict], output_file: str | Path, format: str = 'json'):
        """
        保存问答对到文件

        Args:
            qa_pairs: 问答对列表
            output_file: 输出文件路径
            format: 输出格式 ('json' 或 'txt')
        """
        import json
        # 处理 numpy 类型的转换函数
        def json_default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            return str(o)

        parent_dir = output_file.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    qa_pairs,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=json_default
                )

        elif format == 'txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, pair in enumerate(qa_pairs, 1):
                    f.write(f"=== Question {i} ===\n")
                    f.write(f"Q: {pair['question']}\n")
                    f.write(f"A: {pair['answer']}\n")
                    f.write(f"Source Segment ID: {pair.get('segment_id', 'N/A')}\n")
                    f.write("\n" + "=" * 50 + "\n\n")

        self.logger.info(f"Saved {len(qa_pairs)} question-answer pairs to {output_file}")


def parse_yaml_questions(yaml_path: str, topic: str) -> Dict[str, List[Any]]:
    """
    解析 YAML 文件，返回多个平行列表：
    - questions: 问题文本列表
    - levels: 对应的标题层级名称
    - angles: 对应角度名称
    - difficulties: 难度分数列表
    - explanations: 对应的说明列表
    - answers: 对应的原始答案列表

    :param yaml_path: YAML 文件路径
    :param topic: 要替换的 {topic} 占位符
    :return: 字典，每个 key 对应一个列表
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    questions = []
    levels = []
    level_descriptions = []
    angles = []
    difficulties = []
    explanations = []
    answers = []

    for level in data.get("levels", []):
        level_name = level.get("name")
        level_desc = level.get("description")
        for angle in level.get("angles", []):
            angle_name = angle.get("name")
            for q in angle.get("questions", []):
                question_text = q.get("question", "").replace("{topic}", topic)
                answer_text = q.get("answer", "")
                difficulty = q.get("difficulty", 1)
                explanation = q.get("explanation", "")

                questions.append(question_text)
                levels.append(level_name)
                level_descriptions.append(level_desc)
                angles.append(angle_name)
                difficulties.append(difficulty)
                explanations.append(explanation)
                answers.append(answer_text)

    return {
        "questions": questions,
        "levels": levels,
        "level_descriptions": level_descriptions,
        "angles": angles,
        "difficulties": difficulties,
        "explanations": explanations,
        "answers": answers
    }

