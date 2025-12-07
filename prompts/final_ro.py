all_context_policy = """\
Sunteți un expert în evaluarea paginilor web. Vi se va oferi o pagină web în limba română (text) și va trebui să o clasificați într-una dintre categoriile de mai jos.
De asemenea, vi se va furniza un set de instrucțiuni pentru fiecare categorie.
Va trebui să urmați instrucțiunile și să clasificați pagina web în diferite taxonomii (furnizate mai jos).
"""

topic_policy = """\
Pentru topic, alege una dintre următoarele categorii:

    * Conținut pentru adulți.
    * Artă și design - include: arhitectură.
    * Dezvoltare software - include: algoritmi, programare, web design.
    * Crime și investigații - include: știri despre crime, investigații, rapoarte de poliție. Pentru crime financiare și litigii, alege "Finanțe și afaceri". Pentru probleme sociale sau proces legislativ, alege "Politică și probleme sociale".
    * Educație și joburi - include: știri despre pedagogie, programe de studiu, rapoarte academice. Daca pagina este educativă, dar despre se încadrează la un alt topic (e.g. mâncare sau matematică), alege-l pe acela.
    * Electronică și hardware - include: hardware pentru calculatoare, telefoane, gadgeturi, electronice.
    * Divertisment - include: filme, muzică, seriale, videoclipuri, celebrități, umor, viață de noapte. Daca pagina discută muzică sau filme mai degrabă drept artă decât divertisment, alege "Artă și design".
    * Viață socială - include: familie, prieteni, relații, comunitate. Daca pagina se referă la o activitate socială specifică (e.g. sporturi sau jocuri de societate), alege topic-ul corespunzător activității respective.
    * Modă și frumusețe - include: haine, cosmetică, accesorii.
    * Finanțe și afaceri - include: taxe, regulații, investiții, asigurări, carduri de credit, finanță personală, comunicație corporativă, marketing, resurse umane.
    * Mâncare și băuturi - include: rețete, restaurante, băuturi, alimente. Pentru nutriție (ca știință), alege "Sănătate".
    * Jocuri - include: jocuri video, jocuri de societate, jocuri de noroc.
    * Sănătate - include: medicină, stare de bine, sănătate mentală, sănătate veterinară sau nutriție.
    * Istorie și geografie - include: arheologie
    * Hobby-uri și casă - include: imobiliare, închiriere, relocare, mobilă, îmbunătățiri de casă, DIY, grădinărit, animale de companie, jucării și colecții.
    * Industrial - include: materiale primare, bunuri industriale, substanțe chimice, textile. Topic-uri precum: minerit, agricultură, manufactură, utilități și construcții. Pentru informații generale despre topicuri de afaceri, alege "Finanțe și afaceri".
    * Literatură - include: crtică literară, lingvistică, filosofie și alte subiecte umaniste conexe. Pentru text scris în stil literar, alege topic-ul semnificației (informația transmisă, nu modul de transmitere).
    * Politică - include: probleme sociale, campanii politice, proces legislativ, geopolitică, proteste, activism.
    * Religie - include: spiritualitate
    * Știință, matematică și tehnologie - include: fizică, chimie, biologie, ecologie, matematică, statistică, biotehnologie, inginerie.
    * Software - topicuri legate de software, cum ar fi utilizarea software-ului sau internetul.
    * Sport și fitness - include: arte marțiale, sporturi cu motor, activități în aer liber, echipament sportiv.
    * Transport - include: mașini și alte vehicule, taxiuri, transport public, trafic, naveta, aviație, transport feroviar, livrare, logistică.
    * Turism și călătorii - include: ospitalitate, hoteluri, obiective turistice, croaziere. Pentru descrieri detaliate ale destinațiilor turistice, alege "Geografie și istorie".

Pentru alegera subtopicului, furnizează o specializare (subcategorie a topicului), de maximum 1-4 cuvinte.
"""

format_policy = """\
Pentru format, alege una dintre următoarele categorii:

    * Articol academic - include: lucrari de cercetare, rezumate ale unei lucrări, teze, analize ale literaturii de specialitate. Acest format nu include și paginile web care au academia și cercetarea drept subiect.
    * Cuprins - include: harta site-ului, catalog de produse, rezultate de căutare, liste de știri cu fragmente scurte de articole, director web. Acest format referă la listarea conținutului cu scopul navigării. E posibil ca hyperlinkurile sa fie excluse.
    * Scriere creativă - include: povestiri scurte, capitole dintr-un roman, poezie sau versuri muzicale. Acesta nu include alte pagini web (e.g. forumuri sau articole de știri) care conțin literatură și ficțiune drept topic.
    * Pagină de asistență pentru clienți - include: ghid de depanare. Reprezintă conținut creat de o organizație și destinat publicului larg. Pentru paginile în formatul specific de Întrebări frecvente (FAQ), alege "Întrebări frecvente (FAQs)".
    * Forum de discuții - include: site-uri comunitare precum reddit sau secțiuni de comentarii la articole de știri sau postări de blog. Trebuie să conțină mai multe comentarii.
    * Întrebări frecvente (FAQs) - include: pagini în format Întrebări frecvente.
    * Conținut incomplet - include: pagini cu conținut trunchiat, accesibil doar contra-cost sau care necesită autentificare pentru acces, pagini web multimedia în care textul paginii web descrie și completează în principal conținutul audiovizual, de exemplu o descriere a unui videoclip (YouTube) sau galerie de imagini. Dacă pagina are mai multe fragmente de articole trunchiate, alege "Cuprins".
    * Articol de cunoștințe - include: articole scrise într-un stil obiectiv și neutru, publicate pe o platformă moderată (e.g. Wikipedia) sau de o sursă de încredere.
    * Notificări legale - include: termeni și condiții, declinări legale de răspundere, politică de confidențialitate, acord de licență. Nu include pagini web care au doar subiecte legate de drept.
    * Articol de tip listă - include: bloguri sau articole care prezintă conținutul sub forma unei liste. Exemplu: articole cu "Cele mai bune 10" sau "Top 5 locuri de vizitat în". Dacă lista e menită să ofere o imagine de ansamblu asupra conținutului site-ului și să faciliteze navigarea, alegeți "Cuprins".
    * Articol de știri - include: texte scrise de jurnaliști despre evenimente curente și publicate de organizații de știri. Pentru articole lungi, profiluri, editoriale sau eseuri, alege "Scriere non-ficțiune". Pentru interviuri din presă, alege "Transcriere sau Interviu".
    * Scriere non-ficțiune - include: articole lungi, profiluri, editoriale, necrologuri sau alte forme de scriere non-ficțiune. Acest format referă la scrieri ale unor jurnaliști sau scriitori profesionști.
    * Pagină despre organizație - include: autodescriere sau introducere a unei organizații, cum ar fi o companie, universitate, agenție guvernamentală sau organizație non-profit. Spre deosebire de "Articol de cunoștințe", aceste pagini nu sunt verificate și pot conține auto-promovare.
    * Anunț organizațional - include: comunicate de presă, postare de blog a unei organizații.
    * Pagină personală - pagină "despre" (About) pe un site web personal sau dedicat unui hobby. De obicei, conține o autodescriere, introducere sau informații de profil.
    * Blog personal - articol scris de o persoană, relatând experiențe și opinii personale. Daca secțiunea de comentarii a blogului este mai lungă decât postarea, alegeți "Forum de discuții".
    * Pagină de produs - include: descrieri și promoții pentru un produs sau serviciu (în sens larg, e.g. cursuri universitare). Dacă cea mai mare parte a conținutului constă în recenzii ale utilizatorilor, alegeți "Recenzii ale Utilizatorilor".
    * Forum întrebări și răspunsuri - forum de utilizatori cu un format explicit de întrebări și răspunsuri (e.g. Quora, Stack Exchange).
    * Spam și reclame - pagina conține în principal spam, cuvinte cheie sau reclame scurte pentru alte pagini, produse sau servicii. 
    * Date structurate - intrări multiple de date cu o structură comună (e.g. tabel, foaie de calcul, bază de date de filme, glosar, dicționar, fișiere json, csv, xml).
    * Scriere tehnică - exemple: documentație API, fișiere README, cod sursă. Dacă pagina doar conține un link către documentație, alegeți altă categorie. Spre deosebire de "Pagină de asistență pentru clienți", aceste pagini sunt destinate dezvoltatorilor și experților, mai degrabă decât utilizatorilor finali.
    * Transcriere sau interviu - înregistare scrisă a limbajului vorbit (e.g. interviuri, transcrierea unei audieri în instanță, a unui film, a unui podcast, a unei prelegeri sau a unui discurs).
    * Tutorial sau ghid - include: rețete de gătit, instrucțiuni DIY (do it yourself), pagini WikiHow, cursuri (e.g. Khan Academy). Pe pagină trebuie să apară conținutul real al tutorialului sau ghidului. Dacă pagina conține doar o scurtă descriere sau promovare a tutorialului, alegeți o altă categorie. Dacă ghidul este specific produselor sau serviciilor de pe același site web, alegeți "Pagină de asistență pentru clienți".
    * Recenzii ale utilizatorilor - recenzii postate de utilizatori, de exemplu, pe Yelp sau TripAdvisor.
"""

educational_level_policy = """\
Pentru nivelul de educație la care se adresează, alege una dintre următoarele categorii:
    * Preșcolar - concentrat pe dezvoltarea socială, emoțională cognitivă și fizică prin joc, activități creative și interacțiuni sociale.
    * Școală primară (elementary school)- accent pe alfabetizare, matematică de bază, introducere în știință și dezvoltarea abilităților de bază. 
    * Școală gimnazială (middle school)- se aprofundează materiile din școala primară și se introduc discipline noi (e.g. fizică, chimie, biologie).
    * Liceu (high school)- oferă diverse profiluri (real, uman, tehnologic) și pregătește elevii pentru studii superioare sau pentru intrarea pe piața muncii.
    * Licență (bachelor degree) - primul nivel de studii superioare. Se concentrează pe o disciplină academică sau profesională și oferă cunoștințe de bază și abilități practice.
    * Post-universitar (graduate degree): include studii avansate precum masterat, doctorat, post-doctorat, specializare, rezidențiat. Presupune o pregătire riguroasă într-un domeniu specific.
"""

educational_value_policy = """\
Pentru a alege valoarea educațională (cât de util este textul pentru predare la nivel primar sau gimnazial), folosește următoarea schemă de notare (sistem aditiv cu 5 puncte, punctele sunt acumulate pentru satisfacerea fiecărui criteriu):
    * Adăugați 1 punct dacă extrasul oferă unele informații de bază relevante pentru subiecte educaționale, chiar dacă include conținut irelevant sau non-academic, cum ar fi reclame și materiale promoționale.
    * Adăugați încă un punct dacă extrasul abordează anumite elemente pertinente pentru educație, dar nu se aliniază îndeaproape cu standardele educaționale. Ar putea amesteca conținut educațional cu material non-educațional, oferind o prezentare generală superficială a unor subiecte potențial utile sau prezentând informații într-o manieră dezorganizată și un stil de scriere incoerent.
    * Acordați un al treilea punct dacă extrasul este adecvat pentru uz educațional și introduce concepte cheie relevante pentru programa școlară. Este coerent, deși s-ar putea să nu fie cuprinzător sau ar putea include unele informații irelevante. Poate semăna cu o secțiune introductivă a unui manual sau cu un tutorial de bază care este potrivit pentru învățare, dar are limitări notabile, cum ar fi tratarea unor concepte care sunt prea complexe pentru elevii de gimnaziu.
    * Acordați un al patrulea punct dacă extrasul este foarte relevant și benefic în scopuri educaționale pentru un nivel nu mai mare decât gimnaziul, prezentând un stil de scriere clar și consecvent. Ar putea fi similar cu un capitol dintr-un manual sau un tutorial, oferind conținut educațional substanțial, inclusiv exerciții și soluții, cu informații irelevante minime, iar conceptele nu sunt prea avansate pentru elevii de gimnaziu. Conținutul este coerent, concentrat și valoros pentru învățarea structurată.
    * Acordați un al cincilea punct dacă extrasul este remarcabil prin valoarea sa educațională, perfect potrivit pentru predare fie la școala primară, fie la gimnaziu. Urmează un raționament detaliat, stilul de scriere este ușor de urmărit și oferă perspective profunde și amănunțite asupra subiectului, fără niciun conținut non-educațional sau complex.
"""

all_instructions = """\
Textul extras din pagina web: %s

Mesajul trebuie să respecte următorul format:
    * Explicație: [maxim 200 cuvinte care justifică alegerile făcute; pentru format și topic menționează cuvintele cheie pentru clasificare; pentru valoarea educațională acordată, argumentează ce criterii au fost sau nu îndeplinite și de ce]
    * Valoare educațională: [numărul de puncte acordate, între 0 și 5]
    * Nivel educațional: [unul dintre următoarele: 'Preșcolar', 'Școală primară', 'Școală gimnazială', 'Liceu', 'Licență', 'Post-universitar']
    * Topic: [numele topicului, unul dintre următoarele: 'Conținut pentru adulți', 'Artă și design', 'Dezvoltare software', 'Crime și investigații', 'Educație și joburi', 'Electronică și hardware', 'Divertisment', 'Viață socială', 'Modă și frumusețe', 'Finanțe și afaceri', 'Mâncare și băuturi', 'Jocuri', 'Sănătate', 'Istorie și geografie', 'Hobby-uri și casă', 'Industrial', 'Literatură', 'Politică', 'Religie', 'Știință, matematică și tehnologie', 'Software', 'Sport și fitness', 'Transport', 'Turism și călătorii']
    * Subtopic: [specializare a topic-ului, de maximum 1-4 cuvinte]
    * Format: [numele formatului, unul dintre următoarele: 'Articol academic', 'Cuprins', 'Scriere creativă', 'Pagină de asistență pentru clienți', 'Forum de discuții', 'Întrebări frecvente (FAQs)', 'Conținut incomplet', 'Articol de cunoștințe', 'Notificări legale', 'Articol de tip listă', 'Articol de știri', 'Scriere non-ficțiune', 'Pagină despre organizație', 'Anunț organizațional', 'Pagină personală', 'Blog personal', 'Pagină de produs', 'Forum întrebări și răspunsuri', 'Spam și reclame', 'Date structurate', 'Scriere tehnică', 'Transcriere sau interviu', 'Tutorial sau ghid', 'Recenzii ale utilizatorilor']

Un exemplu de formatare corectă (răspunsul va conține doar aceste informații):
    * Explicație: Textul oferă informații structurate despre nutriție, prezentând avantaje și dezavantaje ale diferitelor diete. Este potrivit pentru elevii de liceu, menționând diverse denumiri tehnice (e.g. macronutrienți). Pentru valoarea educațională, textul indeplinește primele 3 criterii, dar primește punctele de la criteriile 4 și 5 deoarece nu oferă o analiză profundă a subiectului și se adresează unui public mai avansat decât gimnaziul. Principalele cuvinte cheie sunt calorii, dietă, macronutrienți
    * Valoare educațională: 3
    * Nivel educațional: Liceu
    * Topic: Sănătate
    * Subtopic: Nutriție
    * Format: Articol de cunoștințe
"""

all_policy = (
    all_context_policy
    + topic_policy
    + format_policy
    + educational_level_policy
    + educational_value_policy
)


def all_prompt_ro(text: str):
    return [
        {"role": "system", "content": all_policy},
        {"role": "user", "content": all_instructions % text},
    ]


def all_prompt_ro_user(text: str):
    return [
        {"role": "user", "content": all_policy + (all_instructions % text)},
    ]


def all_prompt_ro_gemma(text: str):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": all_policy}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": all_instructions % text}],
        },
    ]


def all_prompt_ro_gemma_without_text():
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": all_policy}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": all_instructions}],
        },
    ]
