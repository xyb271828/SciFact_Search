"""Seeded fictional lesson catalog with topic-held-out queries and graded labels."""
import itertools
import random

from search.data import Dataset, Document

# Public examples composed for this project. No school or student records.
TOPICS = {
    "Math": [
        ("equivalent fractions", "different numerators and denominators for the same amount"),
        ("adding fractions", "combining portions that have unlike bottom numbers"),
        ("coordinate planes", "locating points using horizontal and vertical positions"),
        ("area and perimeter", "measuring space inside and distance around shapes"),
        ("decimal place value", "reading tenths and hundredths in a number"),
    ],
    "Science": [
        ("water cycle", "how moisture moves through air and returns as rain"),
        ("food chains", "how energy moves from plants to animals"),
        ("electric circuits", "how batteries make bulbs light through closed paths"),
        ("forces and motion", "how pushes and pulls change movement"),
        ("rock cycle", "how heating and pressure turn one kind of stone into another"),
    ],
    "English": [
        ("figurative language", "using comparisons and images to describe ideas"),
        ("argument writing", "supporting a claim with reasons and evidence"),
        ("point of view", "identifying who tells a story and what they know"),
        ("main idea", "finding the central message of a passage"),
        ("verb tense", "showing when actions took place"),
    ],
    "Computing": [
        ("loops", "repeating instructions without rewriting them"),
        ("conditionals", "choosing what happens when a rule is true or false"),
        ("binary numbers", "representing amounts with only zeros and ones"),
        ("sorting", "putting a list of items in a specified order"),
        ("debugging", "finding and fixing mistakes in a program"),
    ],
    "Geography": [
        ("map scale", "estimating real distances from lengths on a map"),
        ("latitude and longitude", "locating places using lines around the globe"),
        ("weather and climate", "comparing daily conditions with long-term patterns"),
        ("river systems", "following flowing water from high ground to the sea"),
        ("urban land use", "exploring how cities allocate space for different activities"),
    ],
}
KINDS = ("practice", "lesson", "quiz", "project")
LEVELS = ("beginner", "advanced")
GRADES = range(3, 8)
CONTEXTS = ("at home", "small groups", "independent study", "classroom discussion", "visual examples")


def synthetic_lessons(seed=42):
    rng = random.Random(seed)
    topics = [(subject, topic, alias) for subject, pairs in TOPICS.items() for topic, alias in pairs]
    shuffled = list(range(len(topics)))
    rng.shuffle(shuffled)
    split_by_topic = {
        topic_index: ("train" if i < 15 else "dev" if i < 20 else "test")
        for i, topic_index in enumerate(shuffled)
    }

    docs, by_topic, queries, qrels = {}, {}, {}, {}
    splits = {"train": [], "dev": [], "test": []}
    for i, (subject, topic, alias) in enumerate(topics):
        by_topic[i] = []
        for grade, kind, level, (variant, context) in itertools.product(
            GRADES, KINDS, LEVELS, enumerate(CONTEXTS)
        ):
            did = f"lesson-{i:02}-{grade}-{kind}-{level}-{variant}"
            name = topic if variant % 2 == 0 else alias
            title = f"{name.capitalize()} | {kind.capitalize()}"
            body = f"A {level} Grade {grade} {subject.lower()} resource for {context}."
            docs[did] = Document(title, body, grade, level, kind)
            by_topic[i].append(did)
        for j in range(12):
            grade = GRADES[j % len(GRADES)]
            kind = KINDS[(j + j // 4 + i) % len(KINDS)]
            level = LEVELS[(j // 2) % len(LEVELS)]
            templates = (
                f"Grade {grade} {level} {kind} on {topic}",
                f"I need a {kind} for Year {grade} students who are {level} at {alias}",
                f"{topic} {kind} for grade {grade}",
                f"{kind} about {alias} for {grade}th graders",
            )
            qid = f"lesson-query-{i:02}-{j:02}"
            queries[qid] = templates[j % len(templates)]
            splits[split_by_topic[i]].append(qid)
            judgments = {}
            for did in by_topic[i]:
                doc = docs[did]
                level_matches = doc.level == level or j % 4 >= 2
                if doc.kind == kind and doc.grade == grade and level_matches:
                    judgments[did] = 3
                elif doc.kind == kind and doc.grade == grade:
                    judgments[did] = 2
                elif abs(doc.grade - grade) <= 1:
                    judgments[did] = 1
            qrels[qid] = judgments
    return Dataset(docs, queries, qrels, splits)
