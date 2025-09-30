import numpy as np
import faiss
import logging
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .api_services import *

logger = logging.getLogger(__name__)


@dataclass
class HeadingInfo:
    level: int
    title: str
    start_pos: int
    end_pos: int = None


class SurveyContextBuilder:
    """
    基于问题从Survey文档中提取相关上下文的类
    """

    def __init__(self,
                 embedding_model: EmbeddingService,
                 llm_model: ChatService,
                 dimension: int = DIMENSION,
                 similarity_threshold: float = 0.3,
                 enable_llm_validation: bool = True):
        """
        初始化Survey上下文构建器

        Args:
            embedding_model: 用于向量化的模型
            llm_model: LLM
            similarity_threshold: 相似度阈值
            enable_llm_validation: 是否启用LLM验证步骤
        """
        self.similarity_threshold = similarity_threshold
        self.enable_llm_validation = enable_llm_validation
        self.dimension = dimension
        self.logger = logging.getLogger(__name__)

        # 初始化embedding模型
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # 标题列表List[HeadingInfo]
        self.headings: List[HeadingInfo] = []

        # 标题-正文对
        self.title_text_dict = {}

        # 标题-chunks-faiss
        self.chunk_titles = []

    def drop_ref(self, text):
        marker = "## References"
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].strip()
        marker = "## REFERENCES"
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].strip()
        return text.strip()


    def _extract_headings(self, markdown_text: str) -> List[HeadingInfo]:
        headings = []

        # ATX标题
        atx_pattern = re.compile(r'^(#{1,6})\s*(.*?)(?:\s*#+)?\s*$', re.MULTILINE)
        for match in atx_pattern.finditer(markdown_text):
            level = len(match.group(1))
            title = match.group(2).strip()
            if title:
                headings.append(HeadingInfo(level=level, title=title, start_pos=match.start()))

        # Setext标题
        setext_pattern = re.compile(r'^(.+)\n(=+|-+)\s*$', re.MULTILINE)
        for match in setext_pattern.finditer(markdown_text):
            title = match.group(1).strip()
            underline = match.group(2)
            level = 1 if underline[0] == '=' else 2
            if title:
                headings.append(HeadingInfo(level=level, title=title, start_pos=match.start()))

        # 按位置排序
        headings.sort(key=lambda x: x.start_pos)

        # 设置结束位置
        for i, heading in enumerate(headings):
            if i + 1 < len(headings):
                heading.end_pos = headings[i + 1].start_pos
            else:
                heading.end_pos = len(markdown_text)

        return headings


    def build_title_text_dict(self, markdown_text: str, headings: List[HeadingInfo]) -> Dict[str, str]:
        title_text_dict = {}
        for heading in headings:
            text = markdown_text[heading.start_pos:heading.end_pos].strip()
            # 去掉自身标题行，只保留内容
            lines = text.splitlines()
            if lines:
                title_text_dict[heading.title] = "\n".join(lines[1:]).strip()
            else:
                title_text_dict[heading.title] = ""
        return title_text_dict


    def split_into_chunks(self, text: str, max_words=200) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for para in paragraphs:
            words = para.split()
            start = 0
            while start < len(words):
                chunk_words = words[start:start + max_words]
                chunks.append(" ".join(chunk_words))
                start += max_words
        return chunks


    def process_markdown_for_faiss(self, md_text: str, max_words_per_chunk=200) -> \
    Tuple[Dict[str, str], List[Dict]]:

        # 0, 删去reference信息
        md_text = self.drop_ref(md_text)

        # 1. 提取标题信息
        self.headings = self._extract_headings(md_text)

        # 2. 构建标题 -> 文本字典
        self.title_text_dict = self.build_title_text_dict(md_text, self.headings)

        # 3. 按段落+最大单词数拆分成块
        self.chunk_titles = []
        for title, content in self.title_text_dict.items():
            cur_chunks = self.split_into_chunks(content, max_words_per_chunk)
            if not cur_chunks:
                continue
            self.logger.info("开始构建chunk向量化索引...")
            chunk_vecs = self.embedding_model.get_embedding(
                cur_chunks, batch_size=4, show_progress_bar=True
            ).astype(np.float32)

            # 维度校验
            if chunk_vecs.shape[1] != self.dimension:
                self.logger.error(
                    f"Embedding dimension mismatch: got {chunk_vecs.shape[1]}, expected {self.dimension}"
                )
                continue

            cur_index = faiss.IndexFlatL2(self.dimension)
            cur_index.add(chunk_vecs)

            self.chunk_titles.append({
                "title": title,
                "chunk_contents": cur_chunks,
                "faiss_index": cur_index
            })

        # 如果文档无标题或无有效块
        if not self.chunk_titles:
            self.logger.warning("未找到任何可索引的标题或内容块。")
            return self.title_text_dict, []

        return self.title_text_dict, self.chunk_titles

######################################################################################### 上为建立标题-正文对，标题-chunk-faiss

    def build_heading_hierarchy(self, headings: List[HeadingInfo]) -> Tuple[List[Dict[str, Any]], str]:
        """
        根据 HeadingInfo 列表构建标题层级关系
        返回：
            Tuple[List[Dict], str]：
            - List[Dict]，每个 dict 包含：
                - level: 标题层级
                - title: 标题文本
                - parent_title: 直接父标题，如果没有则为 None
                - path: 从顶层到当前标题的标题列表
                - full_path: path 的字符串表示
                - start_pos: 起始位置
                - end_pos: 结束位置
            - str: LLM友好的大纲描述文本
        """
        hierarchy = []
        stack = []  # 用于跟踪父标题
        outline_lines = []  # 用于构建大纲文本

        for idx, heading in enumerate(headings, 1):
            # 清理栈：移除层级 >= 当前标题的项
            while stack and stack[-1]['level'] >= heading.level:
                stack.pop()

            # 构建路径
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

            # 构建大纲文本行
            # 使用缩进表示层级关系
            indent = "  " * (heading.level - 1)  # 每个层级缩进2个空格
            outline_lines.append(f"{indent}{idx}. {heading.title}")

        # 构建完整的大纲描述
        outline_text = "文档标题大纲：\n" + "\n".join(outline_lines)

        return hierarchy, outline_text

    def _expand_with_sub_titles(self, selected_titles: List[str], valid_titles_set: set) -> List[str]:
        """
        根据选中的标题，自动包含其子标题

        Args:
            selected_titles: LLM选择的初始标题列表
            valid_titles_set: 所有有效标题的集合

        Returns:
            扩展后的标题列表（包含子标题），保证顺序与outline一致
        """
        expanded_titles = set(selected_titles)

        # 构建标题层级映射：{parent_title: [child_titles]}，保持子标题的原始顺序
        parent_to_children = {}
        for heading_info in self.hierarchy:
            parent = heading_info['parent_title']
            current_title = heading_info['title']

            if parent is not None:
                if parent not in parent_to_children:
                    parent_to_children[parent] = []
                parent_to_children[parent].append(current_title)

        # 递归函数：添加所有子标题
        def add_all_children(title: str):
            if title in parent_to_children:
                for child in parent_to_children[title]:
                    if child in valid_titles_set:
                        expanded_titles.add(child)
                        add_all_children(child)  # 递归添加子标题的子标题

        # 对每个选中的标题，添加其所有子标题
        for title in selected_titles:
            add_all_children(title)

        # 确保顺序与outline完全一致：直接按hierarchy顺序过滤
        # 这样可以100%保证顺序正确，因为hierarchy本身就是按原始文档顺序构建的
        ordered_titles = []
        for heading_info in self.hierarchy:
            if heading_info['title'] in expanded_titles:
                ordered_titles.append(heading_info['title'])

        return ordered_titles


    def select_relevant_titles_by_llm(
            self,
            question: str,
            title_text_dict: Dict[str, str],
            chunk_titles: List[Dict],
            include_sub_titles: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        使用 LLM 根据用户问题选择最相关的标题，并返回对应正文内容列表。

        Args:
            question: 用户问题
            title_text_dict: {title: full_text}，来自 process_markdown_for_faiss
            chunk_titles: 每个标题对应的 chunk 和 faiss_index
            include_sub_titles: 是否在高层标题被选中时，自动包含低层标题

        Returns:
            valid_titles: 有效标题列表
            corresponding_texts: 对应正文内容列表（按标题顺序）
        """

        # 1. 构建 outline 文本（每行一个标题，按顺序）
        self.hierarchy, outline_text = self.build_heading_hierarchy(self.headings)

        prompt = f"""
        Based on the following survey outline and the user question, select the most relevant section titles 
        that could contain information to answer the question.

        Guidelines:
        1. Prefer selecting leaf-level headings (titles that do not have any sub-sections) whenever possible.
        2. Ensure that the selected titles together provide enough information to answer the question.
        3. If selecting a high-level heading is necessary to cover the question, include it only when required, 
           but try to select leaf-level headings first.
        4. Avoid including duplicate or irrelevant titles.

        Survey Outline:
        {outline_text}

        User Question:
        {question}

        Please return a JSON list containing only the exact title names from the outline that are most relevant to answering the question.

        Example format:
        ["title1", "title2", "title3"]
        """

        try:
            # 3. 调用 LLM
            message = [{"role": "user", "content": prompt}]
            response = self.llm_model.send_message(message)

            # 4. 尝试提取 JSON 数组，使用更鲁棒的匹配
            # 捕获方括号内的内容，允许换行和空格
            match = re.search(r'\[\s*.*?\s*\]', response, re.DOTALL)
            if not match:
                raise ValueError("未找到 JSON 数组")

            array_str = match.group(0).strip()

            # 5. JSON 解析，尝试多种修复
            try:
                selected_titles = json.loads(array_str)
            except json.JSONDecodeError:
                # 修复单斜杠或非标准引号
                fixed = array_str.replace('\n', ' ')  # 去掉换行
                fixed = re.sub(r'(?<!\\)\\(?![\\"])', r'\\\\', fixed)  # 单斜杠 -> 双
                fixed = fixed.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
                selected_titles = json.loads(fixed)

            # 6. 验证选择的标题是否存在于 chunk_titles
            valid_titles_set = set(item["title"] for item in chunk_titles)
            initial_valid_titles = [t for t in selected_titles if t in valid_titles_set]

            # 7. 新增逻辑：根据include_sub_titles参数决定是否包含子标题
            if include_sub_titles:
                final_titles = self._expand_with_sub_titles(initial_valid_titles, valid_titles_set)
            else:
                final_titles = initial_valid_titles

            # 8. 获取对应正文内容
            corresponding_texts = [title_text_dict[t] for t in final_titles if t in title_text_dict]

            # 9. 打印调试信息
            print("Complete title selection.\n")
            print("valid titles:", final_titles)
            print("\nOriginal selected titles:", selected_titles)
            if include_sub_titles and len(final_titles) > len(initial_valid_titles):
                print(f"Expanded from {len(initial_valid_titles)} to {len(final_titles)} titles (including sub-titles)")

            return final_titles, corresponding_texts

        except Exception as e:
            self.logger.error("Failed o get relevant titles.")
            return [], []


    def _get_title_full_path(self, title: str) -> str:
        """
        获取标题的完整路径
        """
        for heading_info in self.hierarchy:
            if heading_info['title'] == title:
                return heading_info['full_path']
        return title  # 如果找不到，返回原标题

    def get_relevant_paragraphs_from_filtered(
            self,
            question: str,
            selected_titles: List[str] = None,
            top_k_per_title: int = 2,
            similarity_threshold: float = 0.7,
            max_total_paragraphs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        从未被选中的标题对应的text中，选出相关性高的段落作为额外补充信息

        Args:
            question: 用户问题
            selected_titles: 已被选中的标题列表，如果为None则从实例变量获取
            top_k_per_title: 每个标题下最多选择几个段落
            similarity_threshold: 相似度阈值，低于此值的段落不选择
            max_total_paragraphs: 总共最多返回的段落数

        Returns:
            List[Dict]: 相关段落信息列表，每个dict包含：
                - title: 所属标题
                - paragraph: 段落文本
                - similarity_score: 相似度分数
                - chunk_index: 在该标题下的chunk索引
        """

        # 1. 获取已选中的标题集合
        if selected_titles is None:
            # 如果没有提供，尝试从实例变量获取（假设存在last_selected_titles）
            selected_titles = getattr(self, 'last_selected_titles', [])

        selected_titles_set = set(selected_titles)

        # 2. 找出未被选中的标题
        unselected_titles = []
        for chunk_info in self.chunk_titles:
            title = chunk_info["title"]
            if title not in selected_titles_set:
                unselected_titles.append(chunk_info)

        if not unselected_titles:
            print("所有标题都已被选中，无需额外补充。")
            return []

        # 3. 对问题进行向量化
        question_embedding = self.embedding_model.get_embedding(question)
        if question_embedding is None:
            print("无法获取问题的向量表示")
            return []

        # 4. 从每个未选中标题中找出最相关的段落
        candidate_paragraphs = []

        for chunk_info in unselected_titles:
            title = chunk_info["title"]
            chunk_contents = chunk_info["chunk_contents"]
            faiss_index = chunk_info["faiss_index"]

            try:
                # 使用FAISS搜索最相似的chunks
                similarities, indices = faiss_index.search(
                    question_embedding.reshape(1, -1),
                    min(top_k_per_title, len(chunk_contents))
                )

                # 处理搜索结果
                for i, (similarity, chunk_idx) in enumerate(zip(similarities[0], indices[0])):
                    # FAISS返回的是距离，需要转换为相似度
                    # similarity_score = 1 / (1 + similarity)
                    similarity_score = float(1 / (1 + similarity))

                    if similarity_score >= similarity_threshold and chunk_idx < len(chunk_contents):
                        candidate_paragraphs.append({
                            'title': title,
                            'paragraph': chunk_contents[chunk_idx],
                            'similarity_score': similarity_score,
                            'chunk_index': chunk_idx,
                            'full_path': self._get_title_full_path(title)  # 获取完整路径
                        })

            except Exception as e:
                print(f"处理标题 '{title}' 时出错: {e}")
                continue

        # 5. 按相似度排序并选择top-k
        candidate_paragraphs.sort(key=lambda x: x['similarity_score'], reverse=True)
        selected_paragraphs = candidate_paragraphs[:max_total_paragraphs]

        # 6. 打印调试信息
        print(f"\n=== 额外补充段落选择结果 ===")
        print(f"未选中标题数量: {len(unselected_titles)}")
        print(f"候选段落数量: {len(candidate_paragraphs)}")
        print(f"最终选择段落数量: {len(selected_paragraphs)}")

        for i, para in enumerate(selected_paragraphs):
            print(f"\n[补充段落 {i + 1}]")
            print(f"来源标题: {para['title']}")
            print(f"完整路径: {para['full_path']}")
            print(f"相似度分数: {para['similarity_score']:.4f}")
            print(f"段落预览: {para['paragraph'][:100]}...")

        return selected_paragraphs

    def merge_selected_content_with_outline_order(
            self,
            question: str,
            primary_titles: List[str],
            primary_texts: List[str],
            include_supplementary: bool = True,
            supplementary_params: Dict = None
    ) -> List[Dict[str, str]]:
        """
        将主要选中的内容和补充段落按照outline顺序合并，保持原文结构

        Args:
            question: 用户问题
            primary_titles: 主要选中的标题列表
            primary_texts: 主要选中的文本列表
            include_supplementary: 是否包含补充段落
            supplementary_params: 补充段落的参数设置

        Returns:
            List[Dict]: 按outline顺序排列的内容，格式为[{"title": ..., "text": ...}]
        """

        # 1. 构建主要内容的映射
        primary_content_map = dict(zip(primary_titles, primary_texts))

        # 2. 获取补充段落（如果需要）
        supplementary_paragraphs = []
        if include_supplementary:
            if supplementary_params is None:
                supplementary_params = {
                    'top_k_per_title': 2,
                    'similarity_threshold': 0.7,
                    'max_total_paragraphs': 5
                }

            supplementary_paragraphs = self.get_relevant_paragraphs_from_filtered(
                question=question,
                selected_titles=primary_titles,
                **supplementary_params
            )

        # 3. 按标题组织补充段落
        supplementary_by_title = {}
        for para in supplementary_paragraphs:
            title = para['title']
            if title not in supplementary_by_title:
                supplementary_by_title[title] = []
            supplementary_by_title[title].append(para)

        # 对每个标题下的补充段落按相似度排序
        for title in supplementary_by_title:
            supplementary_by_title[title].sort(
                key=lambda x: x['similarity_score'],
                reverse=True
            )

        # 4. 按照outline顺序遍历所有标题，构建最终结果
        result = []
        processed_titles = set()

        for heading_info in self.hierarchy:
            title = heading_info['title']

            # 跳过已处理的标题
            if title in processed_titles:
                continue

            # 检查是否有主要内容
            if title in primary_content_map:
                result.append({
                    "title": title,
                    "text": primary_content_map[title],
                    "source_type": "primary",
                    "full_path": heading_info['full_path']
                })
                processed_titles.add(title)

            # 检查是否有补充段落
            elif title in supplementary_by_title:
                # 将该标题下的所有补充段落合并
                combined_text = self._combine_supplementary_paragraphs(
                    supplementary_by_title[title]
                )

                if combined_text.strip():  # 确保有实际内容
                    result.append({
                        "title": title,
                        "text": combined_text,
                        "source_type": "supplementary",
                        "full_path": heading_info['full_path'],
                        "paragraph_count": len(supplementary_by_title[title]),
                        "avg_similarity": sum(p['similarity_score'] for p in supplementary_by_title[title]) / len(
                            supplementary_by_title[title])
                    })
                    processed_titles.add(title)

        # 5. 打印合并结果统计
        self._print_merge_statistics(result, primary_titles, supplementary_paragraphs)

        return result

    def _combine_supplementary_paragraphs(self, paragraphs: List[Dict]) -> str:
        """
        将同一标题下的多个补充段落合并为一个文本
        """
        if not paragraphs:
            return ""

        if len(paragraphs) == 1:
            return paragraphs[0]['paragraph']

        # 多个段落时，添加分隔和相似度信息
        combined_parts = []
        for i, para in enumerate(paragraphs):
            similarity_note = f"[相关度: {para['similarity_score']:.3f}]"
            combined_parts.append(f"{similarity_note}\n{para['paragraph']}")

        return "\n\n--- 段落分隔 ---\n\n".join(combined_parts)

    def _print_merge_statistics(
            self,
            result: List[Dict],
            primary_titles: List[str],
            supplementary_paragraphs: List[Dict]
    ):
        """
        打印合并结果的统计信息
        """
        primary_count = sum(1 for item in result if item['source_type'] == 'primary')
        supplementary_count = sum(1 for item in result if item['source_type'] == 'supplementary')

        print(f"\n{'=' * 50}")
        print(f"内容合并结果统计")
        print(f"{'=' * 50}")
        print(f"主要选中标题数量: {len(primary_titles)}")
        print(f"补充段落数量: {len(supplementary_paragraphs)}")
        print(f"最终包含标题数量: {len(result)}")
        print(f"  - 主要内容标题: {primary_count}")
        print(f"  - 补充内容标题: {supplementary_count}")

        print(f"\n 最终内容结构:")
        for i, item in enumerate(result, 1):
            source_icon = "Z" if item['source_type'] == 'primary' else "X"
            print(f"{i:2d}. {source_icon} {item['full_path']}")

            if item['source_type'] == 'supplementary':
                para_count = item.get('paragraph_count', 1)
                avg_sim = item.get('avg_similarity', 0)
                print(f"      ↳ {para_count}个段落, 平均相关度: {avg_sim:.3f}")

    # 便捷的一体化函数
    def select_and_merge_content_with_structure(
            self,
            question: str,
            title_text_dict: Dict[str, str],
            chunk_titles: List[Dict],
            include_sub_titles: bool = True,
            include_supplementary: bool = True,
            supplementary_params: Dict = None
    ) -> List[Dict[str, str]]:
        """
        一体化函数：选择内容并按outline结构合并

        这个函数整合了整个流程：
        1. 使用LLM选择主要标题
        2. 获取补充段落
        3. 按outline顺序合并
        """

        # 1. 使用LLM选择主要内容
        primary_titles, primary_texts = self.select_relevant_titles_by_llm(
            question=question,
            title_text_dict=title_text_dict,
            chunk_titles=chunk_titles,
            include_sub_titles=include_sub_titles
        )

        # 2. 合并内容并保持结构
        merged_content = self.merge_selected_content_with_outline_order(
            question=question,
            primary_titles=primary_titles,
            primary_texts=primary_texts,
            include_supplementary=include_supplementary,
            supplementary_params=supplementary_params
        )

        # 3. 转换为简化格式（如果需要）
        simplified_result = []
        for item in merged_content:
            simplified_result.append({
                "title": item["title"],
                "text": item["text"]
            })

        return simplified_result

    # 高级版本：支持更多定制
    def select_and_merge_content_advanced(
            self,
            question: str,
            title_text_dict: Dict[str, str],
            chunk_titles: List[Dict],
            merge_options: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        高级版本：支持更多定制选项

        Args:
            merge_options: 合并选项，包含：
                - include_sub_titles: bool
                - include_supplementary: bool
                - supplementary_params: Dict
                - add_section_markers: bool  # 是否添加章节标记
                - include_metadata: bool     # 是否包含元数据
        """

        if merge_options is None:
            merge_options = {}

        # 默认选项
        default_options = {
            'include_sub_titles': True,
            'include_supplementary': True,
            'supplementary_params': {
                'top_k_per_title': 2,
                'similarity_threshold': 0.7,
                'max_total_paragraphs': 5
            },
            'add_section_markers': False,
            'include_metadata': False
        }

        # 合并选项
        options = {**default_options, **merge_options}

        # 执行选择和合并
        merged_content = self.select_and_merge_content_with_structure(
            question=question,
            title_text_dict=title_text_dict,
            chunk_titles=chunk_titles,
            include_sub_titles=options['include_sub_titles'],
            include_supplementary=options['include_supplementary'],
            supplementary_params=options['supplementary_params']
        )

        # 后处理
        if options['add_section_markers']:
            merged_content = self._add_section_markers(merged_content)

        if not options['include_metadata']:
            # 简化格式
            merged_content = [{"title": item["title"], "text": item["text"]}
                              for item in merged_content]

        return merged_content

    def _add_section_markers(self, content: List[Dict]) -> List[Dict]:
        """
        为内容添加章节标记
        """
        for item in content:
            marker = "=" * 50
            item["text"] = f"{marker}\n{item['title']}\n{marker}\n\n{item['text']}"

        return content


    ########################################################################下为question-answer阶段
    def query_and_answer(self, question: str, simplified_result=None) -> Dict[str, Any]:
        """
        根据问题执行检索，并调用 LLM 返回答案。
        直接使用已构建的 simplified_result 进行搜索。
        返回格式: {"answer": "...", "reference": [...]}
        """
        if not simplified_result:
            raise ValueError("请先调用相关方法构建 simplified_result 数据。")

        # 构建上下文
        context_parts = []
        for i, candidate in enumerate(simplified_result):
            context_parts.append(
                f"[Doc-{i + 1}] Title: {candidate['title']}\n"
                f"Content: {candidate['text']}"
            )
        context = "\n\n".join(context_parts)

        # 构建严格的提示词
        prompt = f"""You are an assistant that strictly answers questions based only on provided documents.

    **CRITICAL RULES:**
    1. You MUST answer ONLY based on the reference documents provided below
    2. You MUST NOT use any external knowledge or make assumptions beyond the documents
    3. If the documents do not contain relevant information, you MUST clearly state "No relevant content found in the documents"
    4. You MUST cite specific document numbers (Doc-1, Doc-2, etc.) when answering

    **Reference Documents:**
    {context}

    **Question:**
    {question}

    **Required Response Format (MUST be valid JSON):**
    {{
      "answer": "Your answer based on the documents, or 'No relevant content found in the documents' if unable to answer",
      "used_docs": [list of document numbers that support your answer, e.g., ["Doc-1", "Doc-3"]],
      "source_text": "The exact text passages from the documents that support your answer"
    }}

    Respond ONLY with the JSON format above, no additional text:"""

        # 调用 LLM
        if not self.llm_model:
            return {
                "answer": "LLM接口未设置",
                "reference": [
                    {
                        "title": candidate["title"],
                        "text": candidate["text"]
                    } for candidate in simplified_result
                ]
            }

        try:
            message = [{"role": "user", "content": prompt}]
            llm_response = self.llm_model.send_message(message)

            # 提取JSON部分（防止LLM添加额外说明）
            # print("\n!!!response is : ", llm_response)
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                response_data = json.loads(json_str)

                # 构建引用信息
                reference_info = []
                used_docs = response_data.get('used_docs', [])

                # 解析使用的文档编号并获取对应信息
                for doc_num in used_docs:
                    try:
                        # 提取文档编号 (Doc-1 -> 0, Doc-2 -> 1, etc.)
                        doc_index = int(doc_num.split('-')[1]) - 1
                        if 0 <= doc_index < len(simplified_result):
                            candidate = simplified_result[doc_index]
                            reference_info.append({
                                "title": candidate["title"],
                                "text": candidate["text"],
                                "doc_number": doc_num
                            })
                    except (IndexError, ValueError):
                        continue

                # 如果没有指定used_docs，就返回所有检索到的内容作为reference
                if not reference_info:
                    reference_info = [
                        {
                            "title": candidate["title"],
                            "text": candidate["text"],
                            "doc_number": f"Doc-{i + 1}"
                        } for i, candidate in enumerate(simplified_result)
                    ]

                return {
                    "answer": response_data.get('answer', 'Failed to parse response'),
                    "reference": reference_info
                }
            else:
                # 如果无法解析JSON，返回原始响应
                return {
                    "answer": llm_response,
                    "reference": [
                        {
                            "title": candidate["title"],
                            "text": candidate["text"],
                            "doc_number": f"Doc-{i + 1}"
                        } for i, candidate in enumerate(simplified_result)
                    ]
                }

        except Exception as e:
            return {
                "answer": f"Error processing response: {str(e)}",
                "reference": [
                    {
                        "title": candidate["title"],
                        "text": candidate["text"],
                        "doc_number": f"Doc-{i + 1}"
                    } for i, candidate in enumerate(simplified_result)
                ]
            }

    def batch_query_and_save(self,
                             questions: List[str],
                             title_text_dict: Dict[str, str],
                             chunk_titles: List[Dict],
                             json_path: str | Path,
                             max_retries: int = 5,
                             delay_between_queries: float = 0.5,
        ) -> Dict[str, Any]:
        """
        批量查询问题并保存结果到JSON文件。
        每次查询都会重新生成上下文收集。

        Args:
            questions: 问题列表
            title_text_dict: 标题到文本的映射字典
            chunk_titles: 文档片段标题列表
            json_path: 保存结果的JSON文件路径
            max_retries: 每个问题的最大重试次数
            delay_between_queries: 查询之间的延迟时间（秒）

        Returns:
            Dict包含统计信息: {
                "metadata": dict,
                "results": List[Dict]
            }
        """
        import json
        import time
        from datetime import datetime

        if not questions:
            raise ValueError("问题列表不能为空")

        parent_dir = Path(json_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        results = []
        successful_count = 0
        failed_count = 0

        print(f"开始批量查询，共 {len(questions)} 个问题...")
        print(f"结果将保存到: {json_path}")

        for i, question in enumerate(questions):
            print(f"\n处理问题 {i + 1}/{len(questions)}: {question[:50]}{'...' if len(question) > 50 else ''}")

            retry_count = 0
            query_successful = False

            while retry_count < max_retries and not query_successful:
                try:
                    # 每次查询都重新生成上下文收集
                    print(f"  正在为问题 {i + 1} 重新生成上下文收集...")

                    supplementary_params = {
                        'top_k_per_title': 2,  # 每个标题最多选2个段落
                        'similarity_threshold': 0.5,  # 相似度阈值
                        'max_total_paragraphs': 5  # 最多5个补充段落
                    }
                    simplified_result = self.select_and_merge_content_with_structure(
                        question=question,
                        title_text_dict=title_text_dict,
                        chunk_titles=chunk_titles,
                        include_sub_titles=True,  # 自动包含子标题
                        include_supplementary=True,  # 包含补充段落
                        supplementary_params=supplementary_params
                    )

                    # 调用单个查询函数
                    answer_result = self.query_and_answer(question, simplified_result)

                    # 构建结果记录
                    result_record = {
                        "question_id": i + 1,
                        "question": question,
                        "answer": answer_result.get("answer", ""),
                        "reference": answer_result.get("reference", []),
                        "context_chunks_count": len(simplified_result) if simplified_result else 0,
                        "timestamp": datetime.now().isoformat(),
                        "retry_count": retry_count
                    }

                    results.append(result_record)
                    successful_count += 1
                    query_successful = True

                    print(f"  查询成功，使用了 {result_record['context_chunks_count']} 个上下文片段")

                except Exception as e:
                    retry_count += 1
                    print(f"  查询失败 (重试 {retry_count}/{max_retries}): {str(e)}")

                    if retry_count >= max_retries:
                        # 记录失败的查询
                        failed_record = {
                            "question_id": i + 1,
                            "question": question,
                            "answer": f"查询失败: {str(e)}",
                            "reference": [],
                            "context_chunks_count": 0,
                            "timestamp": datetime.now().isoformat(),
                            "retry_count": retry_count,
                            "error": str(e)
                        }
                        results.append(failed_record)
                        failed_count += 1
                    else:
                        # 短暂等待后重试
                        time.sleep(delay_between_queries)

            # 查询间延迟（避免API频率限制）
            if i < len(questions) - 1:  # 不是最后一个问题
                time.sleep(delay_between_queries)

        # 构建最终结果 - 所有问题答案都在一个JSON中
        final_result = {
            "metadata": {
                "total_questions": len(questions),
                "successful_queries": successful_count,
                "failed_queries": failed_count,
                "success_rate": successful_count / len(questions) * 100,
                "batch_timestamp": datetime.now().isoformat(),
                "settings": {
                    "max_retries": max_retries,
                    "delay_between_queries": delay_between_queries
                }
            },
            "results": results  # 所有问题的答案都在这个列表中
        }

        # 保存到单个JSON文件
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 所有结果已保存到单个JSON文件: {json_path}")
        except Exception as e:
            print(f"\n✗ 保存文件失败: {str(e)}")
            raise

        # 打印统计信息
        print(f"\n 批量查询完成统计:")
        print(f"   总问题数: {len(questions)}")
        print(f"   成功查询: {successful_count}")
        print(f"   失败查询: {failed_count}")
        print(f"   成功率: {successful_count / len(questions) * 100:.1f}%")

        return final_result

    def load_batch_results(self, json_path: str) -> Dict[str, Any]:
        """
        加载批量查询结果。

        Args:
            json_path: JSON结果文件路径

        Returns:
            加载的结果字典
        """
        import json

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f" 成功加载结果文件: {json_path}")
            return results
        except Exception as e:
            print(f" 加载结果文件失败: {str(e)}")
            raise

    def analyze_batch_results(self, results: Dict[str, Any]) -> None:
        """
        分析批量查询结果。

        Args:
            results: 批量查询结果字典
        """
        metadata = results.get("metadata", {})
        results_list = results.get("results", [])

        print(" 批量查询结果分析:")
        print(f"   总问题数: {metadata.get('total_questions', 0)}")
        print(f"   成功率: {metadata.get('success_rate', 0):.1f}%")
        print(f"   处理时间: {metadata.get('timestamp', 'Unknown')}")

        # 分析失败原因
        failed_results = [r for r in results_list if 'error' in r]
        if failed_results:
            print(f"\n 失败查询分析 ({len(failed_results)} 个):")
            error_types = {}
            for failed in failed_results:
                error_msg = failed.get('error', 'Unknown error')
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                error_types[error_type] = error_types.get(error_type, 0) + 1

            for error_type, count in error_types.items():
                print(f"   - {error_type}: {count} 次")

        # 分析引用数量
        reference_counts = [len(r.get('reference', [])) for r in results_list if 'error' not in r]
        if reference_counts:
            avg_refs = sum(reference_counts) / len(reference_counts)
            print(f"\n 引用统计:")
            print(f"   平均引用数: {avg_refs:.1f}")
            print(f"   最多引用数: {max(reference_counts)}")
            print(f"   最少引用数: {min(reference_counts)}")

    def extract_qa_pairs(self, results: Dict[str, Any], output_path: str = None) -> List[Dict[str, str]]:
        """
        从批量查询结果中提取问答对。

        Args:
            results: 批量查询结果字典
            output_path: 可选的输出文件路径

        Returns:
            问答对列表
        """
        import json

        qa_pairs = []
        results_list = results.get("results", [])

        for result in results_list:
            if 'error' not in result:  # 只提取成功的查询
                qa_pair = {
                    "question": result.get("question", ""),
                    "answer": result.get("answer", ""),
                    "question_id": result.get("question_id", 0)
                }
                qa_pairs.append(qa_pair)

        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
                print(f" 问答对已保存到: {output_path}")
            except Exception as e:
                print(f" 保存问答对失败: {str(e)}")

        print(f" 提取了 {len(qa_pairs)} 个问答对")
        return qa_pairs


def load_json_questions(json_path: str | Path) -> List[str]:
    """
    读取指定 JSON 文件，返回其中所有 'question' 字段构成的列表。

    参数
    ----
    json_path : str | Path
        JSON 文件路径，文件内容应是一个对象数组，每个对象包含 'question' 字段。

    返回
    ----
    List[str]
        question 列表
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 确保是列表结构
    if not isinstance(data, list):
        raise ValueError("JSON 文件内容必须是一个列表")

    return [item["question"] for item in data if "question" in item]
