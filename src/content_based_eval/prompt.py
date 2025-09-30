OUTLINE_COVERAGE_EVAL_PROMPT = \
"""
You are given an outline for a review paper on the topic "{topic}".

Outline:
---
{outline}
---
<instruction>
Please evaluate the outline of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Outline Coverage: Outline Coverage evaluates how comprehensively the outline covers the major aspects, subtopics, and representative directions of the review topic. A high-quality outline should include the core dimensions of the field, reflect current research trends, and avoid missing important areas.
---
Score 1 Description: The outline is largely off-topic or fails to cover the main aspects of the field; most key areas are missing or irrelevant.
Score 2 Description: The outline touches on a few relevant points, but misses many essential subtopics and lacks breadth across the domain.
Score 3 Description: The outline covers some major areas but overlooks several important aspects or emerging directions; coverage is partial.
Score 4 Description: The outline includes most core areas and reasonably reflects the structure of the domain, with only minor omissions.
Score 5 Description: The outline comprehensively covers all major aspects, representative subfields, and recent trends in the topic; no significant omissions.
---
Return the score without any other information:
"""

OUTLINE_RELEVANCE_EVAL_PROMPT = \
"""
You are given an outline for a review paper on the topic "{topic}".

Outline:
---
{outline}
---
<instruction>
Please evaluate the outline of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Outline Relevance: Outline Relevance evaluates how closely the outline aligns with the given topic. A relevant outline should focus on the central research questions, methodologies, and subfields related to the topic, without introducing off-topic or tangential content.
---
Score 1 Description: The outline is mostly unrelated to the given topic; major sections are off-topic or irrelevant.
Score 2 Description: Some sections relate to the topic, but many parts are off-topic or only loosely connected to the core theme.
Score 3 Description: The outline generally aligns with the topic, though several sections deviate from the core focus or include marginally relevant content.
Score 4 Description: Most sections are clearly relevant to the topic, with only minor instances of loosely related material.
Score 5 Description: All sections are highly relevant to the topic, with no noticeable digressions or irrelevant content.
---
Return the score without any other information:
"""


OUTLINE_STRUCTURE_EVAL_PROMPT = \
"""
You are given an outline for a review paper on the topic "{topic}".

Outline:
---
{outline}
---
<instruction>
Please evaluate the outline of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Outline Structure: Outline Structure evaluates the logical organization, hierarchical clarity, and flow of the outline. A well-structured outline should have a clear section-subsection hierarchy, avoid redundancy, and follow a coherent narrative or analytical progression.
---
Score 1 Description: The outline is poorly structured; the hierarchy is unclear or missing, and the sections appear randomly ordered or disconnected.
Score 2 Description: The outline has a basic structure, but many sections are disorganized, improperly nested, or redundant.
Score 3 Description: The outline shows an attempt at organization, with a recognizable hierarchy, but includes some structural issues such as redundancy, inconsistent depth, or unclear flow.
Score 4 Description: The outline is mostly well-organized with a clear hierarchy and logical progression, though it may have minor structural flaws.
Score 5 Description: The outline is clearly structured, with a coherent flow, proper hierarchical depth, and well-balanced section divisions; no noticeable structural issues.
---
Return the score without any other information:
"""

OUTLINE_COVERAGE_EVAL_PROMPT_COMPARE = \
"""
You are given two outlines for a review paper on the topic "{topic}": a human-written reference outline and an AI-generated outline to be evaluated.

Human-written Outline (Reference):
---
{human_outline}
---
AI-generated Outline for Evaluation:
---
{ai_outline}
---
<instruction>
Based on the comparison between the human-written outline and the AI-generated outline about the topic "{topic}", please evaluate the AI outline according to the following criterion. Assign a score from 1 to 5 based on the scoring guidelines.
---
Criterion Description:
Outline Coverage: Outline Coverage evaluates how comprehensively the outline covers the major aspects, subtopics, and representative directions of the review topic. A high-quality outline should include the core dimensions of the field, reflect current research trends, and avoid missing important areas.
---
Score 1 Description: The outline is largely off-topic or fails to cover the main aspects of the field; most key areas are missing or irrelevant.
Score 2 Description: The outline touches on a few relevant points, but misses many essential subtopics and lacks breadth across the domain.
Score 3 Description: The outline covers some major areas but overlooks several important aspects or emerging directions; coverage is partial.
Score 4 Description: The outline includes most core areas and reasonably reflects the structure of the domain, with only minor omissions.
Score 5 Description: The outline comprehensively covers all major aspects, representative subfields, and recent trends in the topic; no significant omissions.
---
Return the score without any other information:
"""

OUTLINE_RELEVANCE_EVAL_PROMPT_COMPARE = \
"""
You are given two outlines for a review paper on the topic "{topic}": a human-written reference outline and an AI-generated outline to be evaluated.

Human-written Outline (Reference):
---
{human_outline}
---
AI-generated Outline for Evaluation:
---
{ai_outline}
---
<instruction>
Based on the comparison between the human-written outline and the AI-generated outline about the topic "{topic}", please evaluate the AI outline according to the following criterion. Assign a score from 1 to 5 based on the scoring guidelines.
---
Criterion Description:
Outline Relevance: Outline Relevance evaluates how closely the outline aligns with the given topic. A relevant outline should focus on the central research questions, methodologies, and subfields related to the topic, without introducing off-topic or tangential content.
---
Score 1 Description: The outline is mostly unrelated to the given topic; major sections are off-topic or irrelevant.
Score 2 Description: Some sections relate to the topic, but many parts are off-topic or only loosely connected to the core theme.
Score 3 Description: The outline generally aligns with the topic, though several sections deviate from the core focus or include marginally relevant content.
Score 4 Description: Most sections are clearly relevant to the topic, with only minor instances of loosely related material.
Score 5 Description: All sections are highly relevant to the topic, with no noticeable digressions or irrelevant content.
---
Return the score without any other information:
"""


OUTLINE_STRUCTURE_EVAL_PROMPT_COMPARE = \
"""
You are given two outlines for a review paper on the topic "{topic}": a human-written reference outline and an AI-generated outline to be evaluated.

Human-written Outline (Reference):
---
{human_outline}
---
AI-generated Outline for Evaluation:
---
{ai_outline}
---
<instruction>
Based on the comparison between the human-written outline and the AI-generated outline about the topic "{topic}", please evaluate the AI outline according to the following criterion. Assign a score from 1 to 5 based on the scoring guidelines.
---
Criterion Description:
Outline Structure: Outline Structure evaluates the logical organization, hierarchical clarity, and flow of the outline. A well-structured outline should have a clear section-subsection hierarchy, avoid redundancy, and follow a coherent narrative or analytical progression.
---
Score 1 Description: The outline is poorly structured; the hierarchy is unclear or missing, and the sections appear randomly ordered or disconnected.
Score 2 Description: The outline has a basic structure, but many sections are disorganized, improperly nested, or redundant.
Score 3 Description: The outline shows an attempt at organization, with a recognizable hierarchy, but includes some structural issues such as redundancy, inconsistent depth, or unclear flow.
Score 4 Description: The outline is mostly well-organized with a clear hierarchy and logical progression, though it may have minor structural flaws.
Score 5 Description: The outline is clearly structured, with a coherent flow, proper hierarchical depth, and well-balanced section divisions; no noticeable structural issues.
---
Return the score without any other information:
"""


CONTENT_COHERENCE_EVAL_PROMPT_CHAPTER = \
"""
Here is the content of a section of a review paper on topic "{topic}".
---
Section Title: "{section_title}"
---
Section Content:
---
{section_content}
---
<instruction>
Please evaluate the content of the section titled "{section_title}", which is part of a review on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Coherence: Content Coherence evaluates the internal logical flow and structural clarity of the section. A coherent section should present its ideas in a clear and orderly manner, with smooth transitions between sentences and paragraphs. The narrative should be consistent, and the section should have a well-defined focus that avoids abrupt shifts or disjointed arguments.
---
Score 1 Description: The section is highly disorganized or incoherent. Sentences and ideas are poorly connected, and the overall flow is confusing or fragmented.
Score 2 Description: The section has some recognizable structure, but the progression of ideas is often unclear. Transitions are weak or missing, resulting in a choppy or uneven narrative.
Score 3 Description: The section generally follows a logical sequence, but there are occasional inconsistencies, unclear transitions, or minor disruptions in the flow of ideas.
Score 4 Description: The section is well-organized and mostly coherent, with logical progression and smooth transitions. Minor improvements could enhance clarity further.
Score 5 Description: The section is highly coherent, with excellent organization, clear transitions, and a consistent, well-structured narrative that enhances readability and understanding.
---
Return the score without any other information:
"""

CONTENT_DEPTH_EVAL_PROMPT_CHAPTER = \
"""
Here is the content of a section of a review paper on topic "{topic}".
---
Section Title: "{section_title}"
---
Section Content:
---
{section_content}
---
<instruction>
Please evaluate the content of the section titled "{section_title}", which is part of a review on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Depth: Content Depth evaluates the analytical richness and intellectual contribution of the section. A high-scoring section should go beyond simple descriptions or listing of works. It should include meaningful comparisons, synthesis of multiple viewpoints, discussion of advantages/limitations, and, where appropriate, offer original insights or highlight research gaps and future directions.
---
Score 1 Description: The section is superficial and lacks meaningful analysis. It merely lists methods or facts without any interpretation, comparison, or commentary.
Score 2 Description: The section includes minimal analytical content. There is occasional mention of advantages or differences, but it lacks depth or consistency.
Score 3 Description: The section demonstrates a basic level of analysis, with some comparisons or interpretations. However, the insights are limited or unevenly developed.
Score 4 Description: The section is generally analytical and demonstrates thoughtful synthesis, with clear comparisons and some original commentary or interpretation.
Score 5 Description: The section is highly analytical and insightful. It effectively synthesizes prior work, identifies patterns or tensions, and offers critical reflections, including open problems or future directions.
---
Return the score without any other information:
"""

CONTENT_FOCUS_EVAL_PROMPT_CHAPTER = \
"""
Here is the content of a section of a review paper on topic "{topic}".
---
Section Title: "{section_title}"
---
Section Content:
---
{section_content}
---
<instruction>
Please evaluate the content of the section titled "{section_title}", which is part of a review on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Focus: Content Focus evaluates how well the section content aligns with the specific topic indicated by its title and the overall topic. A high-scoring section should maintain a clear focus, stay on topic throughout, and avoid including unrelated or off-topic information.
---
Score 1 Description: The section is largely off-topic. Most of the content is irrelevant or only tangentially related to the section title.
Score 2 Description: The section is only loosely related to the stated topic. It includes substantial off-topic discussion or lacks a clear focus.
Score 3 Description: The section addresses the correct topic but occasionally drifts off-topic or includes some loosely related content.
Score 4 Description: The section is mostly focused on the intended topic, with only minor digressions or general statements.
Score 5 Description: The section is tightly focused on the specified topic. All content is clearly relevant and contributes meaningfully to the section’s theme.
---
Return the score without any other information:
"""

CONTENT_COVERAGE_EVAL_PROMPT_CHAPTER = \
"""
Here is the content of a section of a review paper on topic "{topic}".
---
Section Title: "{section_title}"
---
Section Content:
---
{section_content}
---
<instruction>
Please evaluate the content of the section titled "{section_title}", which is part of a review on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Coverage: Content Coverage evaluates the extent to which the section includes the key subtopics, representative works, and essential aspects relevant to its assigned topic. A high-scoring section should comprehensively summarize important developments, methods, or perspectives that are expected in such a section.
---
Score 1 Description: The section omits most major subtopics and fails to mention key works. It reflects very limited awareness of the field.
Score 2 Description: The section covers only a small portion of the expected content. Several important elements are missing or underdeveloped.
Score 3 Description: The section includes some core subtopics or representative works, but the coverage is incomplete or imbalanced.
Score 4 Description: The section addresses most key elements with reasonable detail, though it may lack some breadth or miss minor aspects.
Score 5 Description: The section provides comprehensive coverage of the topic, addressing all major subtopics, representative methods, and important works with appropriate detail.
---
Return the score without any other information:
"""

CONTENT_FLUENCY_EVAL_PROMPT_CHAPTER = \
"""
Here is the content of a section of a review paper on topic "{topic}".
---
Section Title: "{section_title}"
---
Section Content:
---
{section_content}
---
<instruction>
Please evaluate the content of the section titled "{section_title}", which is part of a review on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Fluency: Content Fluency evaluates the clarity, grammatical correctness, and overall readability of the text. A high-scoring section should be well-written, free from grammatical errors, and easy to understand, with smooth transitions between sentences and ideas.
---
Score 1 Description: The text is difficult to read due to frequent grammatical errors, awkward phrasing, or broken sentence structure. Transitions are missing or confusing.
Score 2 Description: The section contains several noticeable grammar or phrasing issues that hinder comprehension. Sentences may be choppy or poorly structured.
Score 3 Description: The section is generally understandable, with some minor grammar or fluency issues. Transitions and sentence flow are adequate but could be improved.
Score 4 Description: The section is well-written with few language issues. Sentences flow smoothly, and transitions are mostly effective.
Score 5 Description: The section is highly fluent and polished. It is clearly written, grammatically correct, and exhibits excellent flow and readability throughout.
---
Return the score without any other information:
"""

CONTENT_COHERENCE_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Score 1 Description: The section is highly disorganized or incoherent. Sentences and ideas are poorly connected, and the overall flow is confusing or fragmented.
Score 2 Description: The section has some recognizable structure, but the progression of ideas is often unclear. Transitions are weak or missing, resulting in a choppy or uneven narrative.
Score 3 Description: The section generally follows a logical sequence, but there are occasional inconsistencies, unclear transitions, or minor disruptions in the flow of ideas.
Score 4 Description: The section is well-organized and mostly coherent, with logical progression and smooth transitions. Minor improvements could enhance clarity further.
Score 5 Description: The section is highly coherent, with excellent organization, clear transitions, and a consistent, well-structured narrative that enhances readability and understanding.
---
Return the score without any other information:
"""

CONTENT_DEPTH_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Score 1 Description: The section is superficial and lacks meaningful analysis. It merely lists methods or facts without any interpretation, comparison, or commentary.
Score 2 Description: The section includes minimal analytical content. There is occasional mention of advantages or differences, but it lacks depth or consistency.
Score 3 Description: The section demonstrates a basic level of analysis, with some comparisons or interpretations. However, the insights are limited or unevenly developed.
Score 4 Description: The section is generally analytical and demonstrates thoughtful synthesis, with clear comparisons and some original commentary or interpretation.
Score 5 Description: The section is highly analytical and insightful. It effectively synthesizes prior work, identifies patterns or tensions, and offers critical reflections, including open problems or future directions.
---
Return the score without any other information:
"""

CONTENT_FOCUS_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Focus: Content Focus evaluates how well the section content aligns with the specific topic indicated by its title and the overall topic. A high-scoring section should maintain a clear focus, stay on topic throughout, and avoid including unrelated or off-topic information.
---
Score 1 Description: The section is largely off-topic. Most of the content is irrelevant or only tangentially related to the section title.
Score 2 Description: The section is only loosely related to the stated topic. It includes substantial off-topic discussion or lacks a clear focus.
Score 3 Description: The section addresses the correct topic but occasionally drifts off-topic or includes some loosely related content.
Score 4 Description: The section is mostly focused on the intended topic, with only minor digressions or general statements.
Score 5 Description: The section is tightly focused on the specified topic. All content is clearly relevant and contributes meaningfully to the section’s theme.
---
Return the score without any other information:
"""

CONTENT_COVERAGE_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Coverage: Content Coverage evaluates the extent to which the section includes the key subtopics, representative works, and essential aspects relevant to its assigned topic. A high-scoring section should comprehensively summarize important developments, methods, or perspectives that are expected in such a section.
---
Score 1 Description: The section omits most major subtopics and fails to mention key works. It reflects very limited awareness of the field.
Score 2 Description: The section covers only a small portion of the expected content. Several important elements are missing or underdeveloped.
Score 3 Description: The section includes some core subtopics or representative works, but the coverage is incomplete or imbalanced.
Score 4 Description: The section addresses most key elements with reasonable detail, though it may lack some breadth or miss minor aspects.
Score 5 Description: The section provides comprehensive coverage of the topic, addressing all major subtopics, representative methods, and important works with appropriate detail.
---
Return the score without any other information:
"""

CONTENT_FLUENCY_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Fluency: Content Fluency evaluates the clarity, grammatical correctness, and overall readability of the text. A high-scoring section should be well-written, free from grammatical errors, and easy to understand, with smooth transitions between sentences and ideas.
---
Score 1 Description: The text is difficult to read due to frequent grammatical errors, awkward phrasing, or broken sentence structure. Transitions are missing or confusing.
Score 2 Description: The section contains several noticeable grammar or phrasing issues that hinder comprehension. Sentences may be choppy or poorly structured.
Score 3 Description: The section is generally understandable, with some minor grammar or fluency issues. Transitions and sentence flow are adequate but could be improved.
Score 4 Description: The section is well-written with few language issues. Sentences flow smoothly, and transitions are mostly effective.
Score 5 Description: The section is highly fluent and polished. It is clearly written, grammatically correct, and exhibits excellent flow and readability throughout.
---
Return the score without any other information:
"""


CONTENT_COHERENCE_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Score 1 Description: The section is highly disorganized or incoherent. Sentences and ideas are poorly connected, and the overall flow is confusing or fragmented.
Score 2 Description: The section has some recognizable structure, but the progression of ideas is often unclear. Transitions are weak or missing, resulting in a choppy or uneven narrative.
Score 3 Description: The section generally follows a logical sequence, but there are occasional inconsistencies, unclear transitions, or minor disruptions in the flow of ideas.
Score 4 Description: The section is well-organized and mostly coherent, with logical progression and smooth transitions. Minor improvements could enhance clarity further.
Score 5 Description: The section is highly coherent, with excellent organization, clear transitions, and a consistent, well-structured narrative that enhances readability and understanding.
---
Return the score without any other information:
"""

CONTENT_DEPTH_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Score 1 Description: The section is superficial and lacks meaningful analysis. It merely lists methods or facts without any interpretation, comparison, or commentary.
Score 2 Description: The section includes minimal analytical content. There is occasional mention of advantages or differences, but it lacks depth or consistency.
Score 3 Description: The section demonstrates a basic level of analysis, with some comparisons or interpretations. However, the insights are limited or unevenly developed.
Score 4 Description: The section is generally analytical and demonstrates thoughtful synthesis, with clear comparisons and some original commentary or interpretation.
Score 5 Description: The section is highly analytical and insightful. It effectively synthesizes prior work, identifies patterns or tensions, and offers critical reflections, including open problems or future directions.
---
Return the score without any other information:
"""

CONTENT_FOCUS_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Focus: Content Focus evaluates how well the section content aligns with the specific topic indicated by its title and the overall topic. A high-scoring section should maintain a clear focus, stay on topic throughout, and avoid including unrelated or off-topic information.
---
Score 1 Description: The section is largely off-topic. Most of the content is irrelevant or only tangentially related to the section title.
Score 2 Description: The section is only loosely related to the stated topic. It includes substantial off-topic discussion or lacks a clear focus.
Score 3 Description: The section addresses the correct topic but occasionally drifts off-topic or includes some loosely related content.
Score 4 Description: The section is mostly focused on the intended topic, with only minor digressions or general statements.
Score 5 Description: The section is tightly focused on the specified topic. All content is clearly relevant and contributes meaningfully to the section’s theme.
---
Return the score without any other information:
"""

CONTENT_COVERAGE_EVAL_PROMPT_ENTIRE_SURVEY = \
"""
Here is the content of a review paper on topic "{topic}".
---
{content}
---
<instruction>
Please evaluate the content of the survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Coverage: Content Coverage evaluates the extent to which the section includes the key subtopics, representative works, and essential aspects relevant to its assigned topic. A high-scoring section should comprehensively summarize important developments, methods, or perspectives that are expected in such a section.
---
Score 1 Description: The section omits most major subtopics and fails to mention key works. It reflects very limited awareness of the field.
Score 2 Description: The section covers only a small portion of the expected content. Several important elements are missing or underdeveloped.
Score 3 Description: The section includes some core subtopics or representative works, but the coverage is incomplete or imbalanced.
Score 4 Description: The section addresses most key elements with reasonable detail, though it may lack some breadth or miss minor aspects.
Score 5 Description: The section provides comprehensive coverage of the topic, addressing all major subtopics, representative methods, and important works with appropriate detail.
---
Return the score without any other information:
"""

CONTENT_FLUENCY_EVAL_PROMPT_COMPARE = \
"""
You are given two surveys for the topic "{topic}": a human-written reference survey and an AI-generated survey to be evaluated.

Human-written Survey (Reference):
---
{human_survey}
---
AI-generated Survey for Evaluation:
---
{ai_survey}
---
<instruction>
Please evaluate the content of the AI-generated survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Fluency: Content Fluency evaluates the clarity, grammatical correctness, and overall readability of the survey. A high-scoring survey should be well-written, free from grammatical errors, and easy to understand, with smooth transitions between sentences and ideas.
---
Score 1 Description: The survey is difficult to read due to frequent grammatical errors, awkward phrasing, or broken sentence structure. Transitions are missing or confusing.
Score 2 Description: The survey contains several noticeable grammar or phrasing issues that hinder comprehension. Sentences may be choppy or poorly structured.
Score 3 Description: The survey is generally understandable, with some minor grammar or fluency issues. Transitions and sentence flow are adequate but could be improved.
Score 4 Description: The survey is well-written with few language issues. Sentences flow smoothly, and transitions are mostly effective.
Score 5 Description: The survey is highly fluent and polished. It is clearly written, grammatically correct, and exhibits excellent flow and readability throughout.
---
Return the score without any other information:
"""

CONTENT_COHERENCE_EVAL_PROMPT_COMPARE = \
"""
You are given two surveys for the topic "{topic}": a human-written reference survey and an AI-generated survey to be evaluated.

Human-written Survey (Reference):
---
{human_survey}
---
AI-generated Survey for Evaluation:
---
{ai_survey}
---
<instruction>
Please evaluate the content of the AI-generated survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Score 1 Description: The survey is highly disorganized or incoherent. Sentences and ideas are poorly connected, and the overall flow is confusing or fragmented.
Score 2 Description: The survey has some recognizable structure, but the progression of ideas is often unclear. Transitions are weak or missing, resulting in a choppy or uneven narrative.
Score 3 Description: The survey generally follows a logical sequence, but there are occasional inconsistencies, unclear transitions, or minor disruptions in the flow of ideas.
Score 4 Description: The survey is well-organized and mostly coherent, with logical progression and smooth transitions. Minor improvements could enhance clarity further.
Score 5 Description: The survey is highly coherent, with excellent organization, clear transitions, and a consistent, well-structured narrative that enhances readability and understanding.
---
Return the score without any other information:
"""

CONTENT_DEPTH_EVAL_PROMPT_COMPARE = \
"""
You are given two surveys for the topic "{topic}": a human-written reference survey and an AI-generated survey to be evaluated.

Human-written Survey (Reference):
---
{human_survey}
---
AI-generated Survey for Evaluation:
---
{ai_survey}
---
<instruction>
Please evaluate the content of the AI-generated survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Score 1 Description: The survey is superficial and lacks meaningful analysis. It merely lists methods or facts without any interpretation, comparison, or commentary.
Score 2 Description: The survey includes minimal analytical content. There is occasional mention of advantages or differences, but it lacks depth or consistency.
Score 3 Description: The survey demonstrates a basic level of analysis, with some comparisons or interpretations. However, the insights are limited or unevenly developed.
Score 4 Description: The survey is generally analytical and demonstrates thoughtful synthesis, with clear comparisons and some original commentary or interpretation.
Score 5 Description: The survey is highly analytical and insightful. It effectively synthesizes prior work, identifies patterns or tensions, and offers critical reflections, including open problems or future directions.
---
Return the score without any other information:
"""

CONTENT_FOCUS_EVAL_PROMPT_COMPARE = \
"""
You are given two surveys for the topic "{topic}": a human-written reference survey and an AI-generated survey to be evaluated.

Human-written Survey (Reference):
---
{human_survey}
---
AI-generated Survey for Evaluation:
---
{ai_survey}
---
<instruction>
Please evaluate the content of the AI-generated survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Focus: Content Focus evaluates how well the survey content aligns with the specific topic indicated by its overall topic. A high-scoring survey should maintain a clear focus, stay on topic throughout, and avoid including unrelated or off-topic information.
---
Score 1 Description: The survey is largely off-topic. Most of the content is irrelevant or only tangentially related to the survey topic.
Score 2 Description: The survey is only loosely related to the stated topic. It includes substantial off-topic discussion or lacks a clear focus.
Score 3 Description: The survey addresses the correct topic but occasionally drifts off-topic or includes some loosely related content.
Score 4 Description: The survey is mostly focused on the intended topic, with only minor digressions or general statements.
Score 5 Description: The survey is tightly focused on the specified topic. All content is clearly relevant and contributes meaningfully to the survey’s theme.
---
Return the score without any other information:
"""

CONTENT_COVERAGE_EVAL_PROMPT_COMPARE = \
"""
You are given two surveys for the topic "{topic}": a human-written reference survey and an AI-generated survey to be evaluated.

Human-written Survey (Reference):
---
{human_survey}
---
AI-generated Survey for Evaluation:
---
{ai_survey}
---
<instruction>
Please evaluate the content of the AI-generated survey on the topic "{topic}", based on the criterion provided below. Assign a score from 1 to 5 according to the scoring guidelines.
---
Criterion Description:
Content Coverage: Content Coverage evaluates the extent to which the survey includes the key subtopics, representative works, and essential aspects relevant to its assigned topic. A high-scoring survey should comprehensively summarize important developments, methods, or perspectives that are expected in such a survey.
---
Score 1 Description: The survey omits most major subtopics and fails to mention key works. It reflects very limited awareness of the field.
Score 2 Description: The survey covers only a small portion of the expected content. Several important elements are missing or underdeveloped.
Score 3 Description: The survey includes some core subtopics or representative works, but the coverage is incomplete or imbalanced.
Score 4 Description: The survey addresses most key elements with reasonable detail, though it may lack some breadth or miss minor aspects.
Score 5 Description: The survey provides comprehensive coverage of the topic, addressing all major subtopics, representative methods, and important works with appropriate detail.
---
Return the score without any other information:
"""