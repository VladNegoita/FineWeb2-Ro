TOPICS = [
    "Conținut pentru adulți",
    "Artă și design",
    "Dezvoltare software",
    "Crime și investigații",
    "Educație și joburi",
    "Electronică și hardware",
    "Divertisment",
    "Viață socială",
    "Modă și frumusețe",
    "Finanțe și afaceri",
    "Mâncare și băuturi",
    "Jocuri",
    "Sănătate",
    "Istorie și geografie",
    "Hobby-uri și casă",
    "Industrial",
    "Literatură",
    "Politică",
    "Religie",
    "Știință, matematică și tehnologie",
    "Software",
    "Sport și fitness",
    "Transport",
    "Turism și călătorii",
]

TOPIC_TO_ID = dict()
for i, topic in enumerate(TOPICS):
    TOPIC_TO_ID[topic] = i

FORMATS = [
    "Articol academic",
    "Cuprins",
    "Scriere creativă",
    "Pagină de asistență pentru clienți",
    "Forum de discuții",
    "Întrebări frecvente (FAQs)",
    "Conținut incomplet",
    "Articol de cunoștințe",
    "Notificări legale",
    "Articol de tip listă",
    "Articol de știri",
    "Scriere non-ficțiune",
    "Pagină despre organizație",
    "Anunț organizațional",
    "Pagină personală",
    "Blog personal",
    "Pagină de produs",
    "Forum întrebări și răspunsuri",
    "Spam și reclame",
    "Date structurate",
    "Scriere tehnică",
    "Transcriere sau interviu",
    "Tutorial sau ghid",
    "Recenzii ale utilizatorilor",
]

FORMAT_TO_ID = dict()
for i, format in enumerate(FORMATS):
    FORMAT_TO_ID[format] = i

AGE_GROUPS = [
    "Preșcolar",
    "Școală primară",
    "Școală gimnazială",
    "Liceu",
    "Licență",
    "Post-universitar",
]

AGE_GROUP_TO_ID = dict()
for i, age_group in enumerate(AGE_GROUPS):
    AGE_GROUP_TO_ID[age_group] = i

SECONDARY_TASKS = {
    "topic": TOPIC_TO_ID,
    "format": FORMAT_TO_ID,
    "age_group": AGE_GROUP_TO_ID,
}

SECONDARY_TASKS_REVERSED = {
    "topic": TOPICS,
    "format": FORMATS,
    "age_group": AGE_GROUPS,
}
