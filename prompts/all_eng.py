general_context = """\
You are a web page reviewer expert.
You will be given a romanian web page (text) and you will have to classify it into one of the categories below.
You will also be given a set of instructions for each category. You will have to follow the instructions and classify the web page into differrent taxonomies (provided below).
"""

topic_policy = """\
Classify the topic of web pages into one of the following 24 categories:

    * Conținut pentru adulți.
    * Artă și design - includes: architecture.
    * Dezvoltare software - includes: algorithms, coding and web development.
    * Crime și investigații - includes: law enforcements; for financial crime and litigation choose 'Finanțe și Afaceri' instead; for social issues and the legislative process, choose 'Politică' instead.
    * Educație și joburi - includes: pedagogy, training & certification, academia; If the page is educational about a specific topic, e.g. food or mathematics, choose that topic instead.
    * Electronică și hardware - includes: computer hardware, phones, televisions, other consumer electronics.
    * Divertisment - includes: music, movies, TV shows, videos, celebrities, humor, nightlife; If the page discusses music or film as art rather than entertainment, choose 'Artă și design' instead.
    * Viață socială - includes: family, friends, relationships, community; If the article focuses on a specific social activity (e.g. sports or board games), choose the topic of the activity instead.
    * Modă și frumusețe - includes: clothing, accessories, cosmetics.
    * Finanțe și afaceri - includes: taxes, regulations, investments, insurance, credit cards, personal finance, corporate communication, marketing, human resources.
    * Mâncare și băuturi - includes: recipes, groceries, beverages, restaurants; for nutritional science, choose 'Sănătate' instead.
    * Jocuri - includes: video games, board games, gambling.
    * Sănătate - includes: medicine, wellness, mental health, veterinary science, nutritional science; for health insurance, choose 'Finanțe și afaceri' instead.
    * Istorie și geografie - includes: archaeology.
    * Hobby-uri și casă - includes: real estate, renting, relocation, furniture, applicanes, home improvement, DIY, gardening, pets, toys, collecting.
    * Industrial - includes: raw materials, industrial goods, chemicals, textiles; topics related to mining, agriculture, manufacturing, utilities and construction; for general business topics or business finance, choose 'Finanțe și afaceri' instead.
    * Literatură - includes: literary criticism, linguistics, philosophy, related subjects in the humanities; for text written in literary style, choose the topic of the contents instead.
    * Politică - includes: social issues, political campaigns, the legislative process, geopolitics, protests, activism.
    * Religie - includes: spirituality.
    * Știință, matematică și tehnologie - includes: physics, chemistry, biology, environmental science, mathematics, statistics, biotech, engineering.
    * Software - includes topics related to the use of software and the internet.
    * Sport și fitness - includes: martial arts, motor sports, outdoor activities, sports equipment.
    * Transport - includes: cars and other vehicles, taxis, public transportation, traffic, commuting, aviation, rail, shipping, logistics.
    * Turism și călătorii - includes: hospitality, hotels, sight-seeing, cruises; for detailed descriptions of tourist destinations, choose 'Istorie și geografie' instead.

Additionally, add a subtopic (this one can be anything, usually a sub-category of the general topic).
"""

format_policy = """\
Classify the format of web pages into one of the following 24 categories:

    * Articol academic - includes: a research paper, a paper abstract, a thesis, a literature review; this does not include other web pages that have academia and research only as their topic.
    * Cuprins - includes: sitemap, product catalog, search results, news listings with short snippets of articles, web directory; the page contains an overview of content and is used for navigation; Note that hyperlinks are not visible from the text content and you have to deduce which parts of the page contain hyperlinks.
    * Scriere creativă - includes: a short story, chapters from a novel, a poem or song lyrics; this does not include other web pages (e.g. forums or news articles) that have literature and fiction only as their topic.
    * Pagină de asistență pentru clienți - example: a troubleshooting guide; content by an organization and for a general audience; for customer support pages in the specific format of FAQs, choose 'Întrebări frecvente (FAQs)' instead.
    * Forum de discuții - includes: Community sites like reddit or comment sections on news article or blog posts; has to contain multiple posts or comments.
    * Întrebări frecvente (FAQs) - the page content is in the Frequently Asked Questions format.
    * Conținut incomplet - includes: page contents that are truncated, pay-walled, require a login to access or multimedia web pages where the web page text primarily describes and supplements the audiovisual content, e.g., a YouTube video description or image gallery; if the page has multiple snippets of truncated articles, choose 'Curpins' instead.
    * Articol de cunoștințe - includes: articles written in an objective and neutral style, published on a moderated platform (like Wikipedia) or by a reputable source.
    * Notificări legale - includes: terms of service, legal disclaimers, privacy policy, license agreement; this does not include other web pages that only have law-related topics.
    * Articol de tip listă - includes: blog or article that presents content in the form of a list; examples: Buzzfeed-style articles, "Top 10" lists, "7 things you didn't know about Y", "4 best places to visit in Z"; if the list is meant to give an overview of the site contents and facilitate navigation, choose 'Cuprins' instead.
    * Articol de știri - includes: text written by journalists on current events and published by news organizations; for long reads, profiles, editorials, and journalistic essays, choose 'Scriere non-ficțiune' instead; for newspaper interviews, choose 'Transcriere sau interviu' instead.
    * Scriere non-ficțiune - includes: long reads, profiles, editorials, essays, obituaries, memoirs and other forms of nonfiction writing; written by journalists and other professional writers.
    * Pagină despre organizație - typically contains a self-description or introduction by an organization such as a company, university, government agency, non-profit organization; note that the content may appear similar to a 'Articol de cunoștințe' in some cases, but is not verified and may contain self-promotion.
    * Anunț organizațional - includes: a press release, a blog post by an organization such as a company, university, government agency, non-profit organization.
    * Pagină personală - an "About Page" on a personal website or hobby website; typically contains a self-description, introduction or profile information.
    * Blog personal - written by an individual typically relating personal experiences and opinions; f the blog's comment section is longer than the blog post, choose 'Forum de discuții' instead.
    * Pagină de produs - includes: descriptions and promotions for a product, service or products in a wider sense, for example university course descriptions; if most of the page content consists of user reviews, choose 'Recenzii ale utilizatorilor' instead.
    * Forum întrebări și răspunsuri - includes: user forum with an explicit question & answer format, e.g., Quora, Stack Exchange.
    * Spam și reclame - includes: pages that consists primarily of spam content, SEO keyword stuffing, or short online ads for other pages, products or services; also choose this category if the page has no apparent purpose.
    * Date structurate - includes: table, datasheet, movie database, glossary, dictionary, json file, csv, xml; multiple data entries with a common structure.
    * Scriere tehnică - includes: API documentation, README files, source code; if the page only contains a link to documentation, choose a different category instead; unlike 'Pagină de asistență pentru clienți', this content is meant for developers and experts, rather than end-users.
    * Transcriere sau interviu - includes: written record of spoken language; examples: interviews (e.g. in a newspaper), the transcript of a court hearing, movie, podcast, lecture or speech.
    * Tutorial sau ghid - includes: cooking recipes, DIY instructions, a WikiHow page, a Khan Academy course; the page must contain the actual content of the tutorial / how-to guide; if the page only contains a brief description or promotion of the tutorial, choose a different category instead; if the guide is specific to products or services from the same website, choose 'Pagină de asistență pentru clienți' instead.
    * Recenzii ale utilizatorilor - includes: reviews posted by users, e.g., on Yelp, TripAdvisor.
"""

educational_level_policy = """\
Classify the educational level of web pages into one of the following 6 categories:

    * Preșcolar - focused on social, emotional, cognitive, and physical development through play, creative activities, and social interactions.
    * Școală primară (elementary school) - emphasis on literacy, basic mathematics, introduction to science, and the development of fundamental skills.
    * Școală gimnazială (middle school) - deepens the subjects from primary school and introduces new disciplines (e.g., physics, chemistry, biology).
    * Liceu (high school) - offers various profiles (science, humanities, technology) and prepares students for higher education or entry into the workforce.
    * Licență (bachelor degree) - the first level of higher education. It focuses on an academic or professional discipline and provides basic knowledge and practical skills.
    * Post-universitar (graduate degree) - includes advanced studies such as master's, doctoral, post-doctoral, specialization, or residency, requiring rigorous training in a specific field.
"""

educational_value_policy = """\
Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
    * Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
    * Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
    * Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. 
    * Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
    * Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.
"""

all_instructions = """\
The extracted text from the web page: %s

The message should follow the following format:
    * Explicație: [maximum 200 words that justify the choices made; for format and topic mark the important keywords that are relevant for the classification; for the educational value argument which criteria were met and which were not met and why]
    * Valoare educațională: [a number between 0 and 5]
    * Nivel educațional: [one of 'Preșcolar', 'Școală primară', 'Școală gimnazială', 'Liceu', 'Licență', 'Post-universitar']
    * Topic: [one of the provided topics: 'Conținut pentru adulți', 'Artă și design', 'Dezvoltare software', 'Crime și investigații', 'Educație și joburi', 'Electronică și hardware', 'Divertisment', 'Viață socială', 'Modă și frumusețe', 'Finanțe și afaceri', 'Mâncare și băuturi', 'Jocuri', 'Sănătate', 'Istorie și geografie', 'Hobby-uri și casă', 'Industrial', 'Literatură', 'Politică', 'Religie', 'Știință, matematică și tehnologie', 'Software', 'Sport și fitness', 'Transport' sau 'Turism și călătorii']
    * Subtopic [any subtopic]
    * Format: [one of the provided formats: 'Articol academic', 'Cuprins', 'Scriere creativă', 'Pagină de asistență pentru clienți', 'Forum de discuții', 'Întrebări frecvente (FAQs)', 'Conținut incomplet', 'Articol de cunoștințe', 'Notificări legale', 'Articol de tip listă', 'Articol de știri', 'Scriere non-ficțiune', 'Pagină despre organizație', 'Anunț organizațional', 'Pagină personală', 'Blog personal', 'Pagină de produs', 'Forum întrebări și răspunsuri', 'Spam și reclame', 'Date structurate', 'Scriere tehnică', 'Transcriere sau interviu', 'Tutorial sau ghid' sau 'Recenzii ale utilizatorilor']
    
Here is an example of a correct output:
    * Explicație: Textul oferă informații structurate despre nutriție, prezentând avantaje și dezavantaje ale diferitelor diete. Este potrivit pentru elevii de liceu, menționând diverse denumiri tehnice (e.g. macronutrienți). Pentru valoarea educațională, textul indeplinește primele 3 criterii, dar primește punctele de la criteriile 4 și 5 deoarece nu oferă o analiză profundă a subiectului și se adresează unui public mai avansat decât gimnaziul. Principalele cuvinte cheie sunt calorii, dietă, macronutrienți
    * Valoare educațională: 3
    * Nivel educațional: Liceu
    * Topic: Sănătate
    * Subtopic: Nutriție
    * Format: Articol de cunoștințe
"""


def all_prompt(text: str):
    return [
        {
            "role": "system",
            "content": general_context
            + topic_policy
            + format_policy
            + educational_level_policy
            + educational_value_policy,
        },
        {"role": "user", "content": all_instructions % text},
    ]


educational_value_policy_finewebedu = """\
Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. 
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
%s.

After examining the extract: 
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "* Valoare educațională: <total points>"
"""


def educational_value_prompt(text: str):
    return [
        {"role": "user", "content": (educational_value_policy_finewebedu % text)},
    ]
