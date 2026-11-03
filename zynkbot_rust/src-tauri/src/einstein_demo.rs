//! The Einstein demo: a memory set a new user can load to see what Zynkbot does with
//! what it is told, written the way the memory system of 2026-10-24 would have stored
//! it. Each entry is one message Einstein might have typed (first person, `said`) and
//! the facts the extractor would have kept from it (third person, starting with the
//! user's name, one thing with its details per fact). The dates are when the thing
//! happened, so the Memory Manager reads as a life in order. The links are the four
//! kinds the app makes: "related", "updates" (the older memory becomes history),
//! "contradicts", and "resolves" (the user's own explanation of a conflict).
//!
//! Before 2026-11-03 the demo was a SQL file from the first memory system (paragraph
//! memories, six link types); 98 of its 101 links failed the four-type check that
//! migration 0022 added, so it loaded as 59 memories with three links.

use sqlx::SqlitePool;

pub const SESSION_ID: &str = "einstein-demo-session";
pub const NAME: &str = "Albert";

pub struct Fact {
    pub key: &'static str,
    pub title: &'static str,
    pub fact: &'static str,
    pub namespace: &'static str,
    pub tags: &'static [&'static str],
    /// (name as written, kind: person | place | org | thing)
    pub entities: &'static [(&'static str, &'static str)],
}

pub struct Message {
    pub date: &'static str,
    pub said: &'static str,
    pub facts: &'static [Fact],
}

pub struct Link {
    pub source: &'static str,
    pub target: &'static str,
    pub relation: &'static str,
    pub confidence: f32,
    pub note: &'static str,
}

macro_rules! fact {
    ($key:expr, $title:expr, $fact:expr, $ns:expr, [$($tag:expr),*], [$(($e:expr, $k:expr)),*]) => {
        Fact { key: $key, title: $title, fact: $fact, namespace: $ns, tags: &[$($tag),*], entities: &[$(($e, $k)),*] }
    };
}

pub static MESSAGES: &[Message] = &[
    Message { date: "1879-03-14", said: "I was born on March 14, 1879, in Ulm, Germany. My parents were Hermann Einstein and Pauline Koch.", facts: &[
        fact!("birth", "Born in Ulm, 14 March 1879", "Albert was born on March 14, 1879, in Ulm, Germany.", "biography", ["birth", "ulm"], [("Ulm", "place"), ("Germany", "place")]),
        fact!("parents", "Parents Hermann Einstein and Pauline Koch", "Albert's parents were Hermann Einstein and Pauline Koch.", "family", ["parents"], [("Hermann Einstein", "person"), ("Pauline Koch", "person")]),
    ]},
    Message { date: "1882-06-01", said: "I didn't speak until I was 3 years old, which worried my parents. I developed the habit of repeating sentences to myself before saying them out loud.", facts: &[
        fact!("speech", "Did not speak until age 3", "Albert didn't speak until he was 3 years old, which worried his parents, and he developed the habit of repeating sentences to himself before saying them out loud.", "biography", ["childhood", "speech"], []),
    ]},
    Message { date: "1884-12-01", said: "When I was 5, my father showed me a compass. The invisible force that made the needle move fascinated me and sparked my lifelong interest in physics.", facts: &[
        fact!("compass", "The compass at age 5", "When Albert was 5, his father showed him a compass, and the invisible force that moved the needle sparked his lifelong interest in physics.", "biography", ["childhood", "compass", "physics"], []),
    ]},
    Message { date: "1887-09-01", said: "My mother, Pauline Koch, was a talented pianist. She made sure I learned violin from age 6. I've played it ever since and named my violin 'Lina'.", facts: &[
        fact!("mother", "Mother Pauline, a talented pianist", "Albert's mother Pauline Koch was a talented pianist.", "family", ["mother", "music"], [("Pauline Koch", "person")]),
        fact!("violin", "Learned violin from age 6, named it Lina", "Albert learned the violin from age 6 at his mother's insistence, has played it ever since, and named his violin 'Lina'.", "hobbies", ["violin", "music"], [("Lina", "thing")]),
    ]},
    Message { date: "1896-10-29", said: "After initially failing the entrance exam, I have been admitted to the Swiss Federal Polytechnic School, ETH Zurich.", facts: &[
        fact!("eth", "Admitted to ETH Zurich, 1896", "Albert was admitted to the Swiss Federal Polytechnic School (ETH Zurich) in 1896 after initially failing the entrance exam.", "education", ["eth", "zurich"], [("ETH Zurich", "org")]),
    ]},
    Message { date: "1901-02-21", said: "I became a citizen of Switzerland today.", facts: &[
        fact!("swiss", "Became a Swiss citizen, 1901", "Albert became a citizen of Switzerland in 1901.", "biography", ["citizenship", "switzerland"], [("Switzerland", "place")]),
    ]},
    Message { date: "1902-06-16", said: "I have started work as a patent examiner at the patent office in Bern, Switzerland. The job leaves me time to think deeply about physics.", facts: &[
        fact!("patent", "Patent examiner in Bern", "Albert works as a patent examiner at the patent office in Bern, Switzerland, a job that leaves him time to think deeply about physics.", "career", ["patent office", "bern"], [("Bern", "place"), ("Switzerland", "place")]),
    ]},
    Message { date: "1902-09-01", said: "My friends and I have formed the 'Olympia Academy' here in Bern to discuss philosophy, science and literature.", facts: &[
        fact!("olympia", "The Olympia Academy in Bern", "Albert and his friends formed the 'Olympia Academy' in Bern to discuss philosophy, science and literature.", "hobbies", ["olympia academy", "friends"], [("Olympia Academy", "org"), ("Bern", "place")]),
    ]},
    Message { date: "1903-01-06", said: "I married Mileva Marić today. She is a fellow physics student.", facts: &[
        fact!("mileva", "Married Mileva Marić, January 1903", "Albert married Mileva Marić, a fellow physics student, in January 1903.", "family", ["marriage", "mileva"], [("Mileva Marić", "person")]),
    ]},
    Message { date: "1910-07-28", said: "Mileva and I have three children: Lieserl, Hans Albert and Eduard, who was born today.", facts: &[
        fact!("children", "Three children with Mileva", "Albert and Mileva Marić have three children: Lieserl, Hans Albert and Eduard.", "family", ["children"], [("Mileva Marić", "person"), ("Lieserl", "person"), ("Hans Albert", "person"), ("Eduard", "person")]),
    ]},
    Message { date: "1905-01-15", said: "My patent work involves evaluating electromagnetic devices, which has shaped how I think about light and relativity.", facts: &[
        fact!("patent_light", "Patent work shaped his thinking on light", "Albert's patent examination work involves evaluating electromagnetic devices, which has shaped how he thinks about light and relativity.", "career", ["patent office", "light"], []),
    ]},
    Message { date: "1905-03-18", said: "I have submitted a paper on the photoelectric effect proposing that light consists of discrete quanta, or photons.", facts: &[
        fact!("photoelectric", "The photoelectric effect paper, 1905", "Albert's 1905 paper on the photoelectric effect proposed that light consists of discrete quanta (photons).", "science", ["photoelectric effect", "quanta"], [("photoelectric effect", "thing")]),
    ]},
    Message { date: "1905-05-11", said: "My paper on Brownian motion gives empirical evidence for the existence of atoms and molecules.", facts: &[
        fact!("brownian", "The Brownian motion paper, 1905", "Albert's 1905 paper on Brownian motion provided empirical evidence for the existence of atoms and molecules.", "science", ["brownian motion", "atoms"], [("Brownian motion", "thing")]),
    ]},
    Message { date: "1905-06-30", said: "My paper 'On the Electrodynamics of Moving Bodies' introduces special relativity. Time and space are relative, not absolute. The speed of light in a vacuum is the same regardless of the motion of the source or the observer, and time passes more slowly for objects moving at high speed relative to a stationary observer.", facts: &[
        fact!("sr", "Special relativity, 1905", "Albert's 1905 paper 'On the Electrodynamics of Moving Bodies' introduced special relativity, showing that time and space are relative, not absolute.", "science", ["special relativity"], [("special relativity", "thing")]),
        fact!("light", "The speed of light is constant", "Albert postulated that the speed of light in a vacuum is the same regardless of the motion of the light source or the observer, the founding postulate of special relativity.", "science", ["special relativity", "light"], [("special relativity", "thing")]),
        fact!("dilation", "Time dilation", "Albert found that time passes more slowly for objects moving at high speed relative to a stationary observer, a consequence of special relativity.", "science", ["special relativity", "time"], [("special relativity", "thing")]),
    ]},
    Message { date: "1905-09-27", said: "I have derived the equation E=mc², showing that energy and mass are interchangeable.", facts: &[
        fact!("emc2", "E=mc²", "Albert derived the equation E=mc², showing that energy and mass are interchangeable.", "science", ["e=mc²", "energy"], [("E=mc²", "thing")]),
    ]},
    Message { date: "1905-12-01", said: "This year I published four papers in Annalen der Physik: the photoelectric effect, Brownian motion, special relativity and mass-energy equivalence. People have started calling it my miracle year.", facts: &[
        fact!("miracle", "The miracle year, 1905", "In 1905 Albert published four groundbreaking papers in Annalen der Physik, a year that became known as his 'miracle year'.", "achievements", ["miracle year", "1905"], [("Annalen der Physik", "org")]),
    ]},
    Message { date: "1907-11-01", said: "I have realized that gravitational acceleration is indistinguishable from acceleration due to mechanical forces. I call this the equivalence principle.", facts: &[
        fact!("equivalence", "The equivalence principle", "Albert realized that gravitational acceleration is indistinguishable from acceleration due to mechanical forces, which became the equivalence principle.", "science", ["equivalence principle", "gravity"], [("equivalence principle", "thing")]),
    ]},
    Message { date: "1911-06-21", said: "I predict that massive objects bend the paths of light, which should produce gravitational lensing effects observable in astronomy.", facts: &[
        fact!("lensing", "Predicted gravitational lensing", "Albert predicted that massive objects bend light paths, creating gravitational lensing effects observable in astronomy.", "science", ["gravitational lensing"], [("gravitational lensing", "thing")]),
    ]},
    Message { date: "1913-06-29", said: "I now describe gravity not as a force but as the curvature of four-dimensional spacetime by massive objects.", facts: &[
        fact!("spacetime", "Gravity as the curvature of spacetime", "Albert describes gravity not as a force but as the curvature of four-dimensional spacetime by massive objects.", "science", ["spacetime", "gravity"], [("spacetime", "thing")]),
    ]},
    Message { date: "1914-04-06", said: "I have accepted a professorship at the Prussian Academy of Sciences in Berlin.", facts: &[
        fact!("berlin", "Professor in Berlin", "Albert is a professor at the Prussian Academy of Sciences in Berlin.", "career", ["berlin", "professor"], [("Prussian Academy of Sciences", "org"), ("Berlin", "place")]),
    ]},
    Message { date: "1914-09-01", said: "I am a committed pacifist. I oppose militarism and nationalism.", facts: &[
        fact!("pacifism", "A committed pacifist", "Albert is a committed pacifist who opposes militarism and nationalism.", "politics", ["pacifism"], []),
    ]},
    Message { date: "1914-10-04", said: "I am one of very few German intellectuals to publicly oppose this war. I have signed the anti-war manifesto.", facts: &[
        fact!("wwi", "Publicly opposed World War I", "Albert is one of few German intellectuals to publicly oppose World War I, and has signed anti-war manifestos.", "politics", ["world war i", "pacifism"], [("World War I", "thing")]),
    ]},
    Message { date: "1915-11-25", said: "Between 1907 and now I have developed general relativity, which describes gravity as the curvature of spacetime caused by mass and energy.", facts: &[
        fact!("gr", "General relativity, 1915", "Between 1907 and 1915 Albert developed general relativity, describing gravity as the curvature of spacetime caused by mass and energy.", "science", ["general relativity"], [("general relativity", "thing")]),
    ]},
    Message { date: "1916-01-13", said: "My field equations predict regions where spacetime curvature becomes so extreme that nothing can escape: black holes.", facts: &[
        fact!("blackholes", "Black holes from the field equations", "Albert's general relativity equations predict black holes, regions where spacetime curvature is so extreme that nothing can escape.", "science", ["black holes", "general relativity"], [("black holes", "thing"), ("general relativity", "thing")]),
    ]},
    Message { date: "1916-06-22", said: "I predict gravitational waves: ripples in spacetime caused by accelerating masses.", facts: &[
        fact!("gravwaves", "Predicted gravitational waves, 1916", "In 1916 Albert predicted gravitational waves, ripples in spacetime caused by accelerating masses.", "science", ["gravitational waves"], [("gravitational waves", "thing")]),
    ]},
    Message { date: "1917-02-08", said: "I have introduced a cosmological constant into my equations so that they allow a static universe.", facts: &[
        fact!("cosmological", "Introduced the cosmological constant", "In 1917 Albert introduced the cosmological constant into his equations to allow a static universe.", "science", ["cosmological constant"], [("cosmological constant", "thing")]),
    ]},
    Message { date: "1919-02-14", said: "Mileva and I have divorced. I am marrying my cousin Elsa Löwenthal this year.", facts: &[
        fact!("divorce", "Divorced Mileva, 1919", "Albert and Mileva Marić divorced in 1919.", "family", ["divorce", "mileva"], [("Mileva Marić", "person")]),
        fact!("elsa", "Married Elsa Löwenthal, 1919", "Albert married his cousin Elsa Löwenthal in 1919.", "family", ["marriage", "elsa"], [("Elsa Löwenthal", "person")]),
    ]},
    Message { date: "1919-11-07", said: "Arthur Eddington's eclipse expedition has confirmed my prediction that massive objects bend light. I have become world-famous overnight.", facts: &[
        fact!("eclipse", "The 1919 eclipse confirmed light bending", "Arthur Eddington's 1919 solar eclipse expedition confirmed Albert's prediction that massive objects bend light, and Albert became world-famous overnight.", "achievements", ["eclipse", "fame"], [("Arthur Eddington", "person")]),
    ]},
    Message { date: "1920-03-15", said: "I'm a passionate pipe smoker. Smoking contributes to a calm and objective judgment in all human affairs.", facts: &[
        fact!("pipe", "A passionate pipe smoker", "Albert is a passionate pipe smoker and believes smoking contributes to a calm and objective judgment in all human affairs.", "personal", ["pipe", "smoking"], []),
    ]},
    Message { date: "1921-04-02", said: "I support Zionism, and I am travelling to America to raise funds for the Hebrew University in Jerusalem.", facts: &[
        fact!("zionism", "Supports Zionism", "Albert supports Zionism and is travelling to America to raise funds for the Hebrew University in Jerusalem.", "politics", ["zionism"], [("Zionism", "thing"), ("America", "place"), ("Hebrew University", "org"), ("Jerusalem", "place")]),
    ]},
    Message { date: "1921-11-09", said: "I have been awarded the 1921 Nobel Prize in Physics, for my discovery of the law of the photoelectric effect, not for relativity.", facts: &[
        fact!("nobel", "Nobel Prize in Physics, 1921", "Albert won the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect, not for relativity.", "achievements", ["nobel prize"], [("Nobel Prize", "thing"), ("photoelectric effect", "thing")]),
    ]},
    Message { date: "1924-12-01", said: "I have extended Satyendra Nath Bose's work on photon statistics to massive particles. It predicts a new state of matter, the Bose-Einstein condensate.", facts: &[
        fact!("bose", "Bose-Einstein statistics", "Albert extended Satyendra Nath Bose's work on photon statistics to massive particles, predicting Bose-Einstein condensates.", "science", ["quantum statistics", "bose"], [("Satyendra Nath Bose", "person"), ("Bose-Einstein condensate", "thing")]),
    ]},
    Message { date: "1927-10-24", said: "I helped found quantum mechanics, but I cannot accept that nature is fundamentally probabilistic. God does not play dice with the universe. I believe in a deterministic universe.", facts: &[
        fact!("qm", "Skeptical of quantum mechanics", "Despite helping found quantum mechanics, Albert remains skeptical of its probabilistic nature, saying 'God does not play dice'.", "philosophy", ["quantum mechanics"], [("quantum mechanics", "thing")]),
        fact!("determinism", "Believes in a deterministic universe", "Albert believes in a deterministic universe.", "philosophy", ["determinism"], []),
    ]},
    Message { date: "1927-10-30", said: "Niels Bohr and I have been debating the interpretation of quantum mechanics all week at the Solvay conference. We disagree fundamentally.", facts: &[
        fact!("bohr", "Debates with Niels Bohr", "Albert and Niels Bohr debate the interpretation of quantum mechanics, most recently at the Solvay conference, and disagree fundamentally.", "philosophy", ["bohr", "quantum mechanics"], [("Niels Bohr", "person"), ("Solvay conference", "org"), ("quantum mechanics", "thing")]),
    ]},
    Message { date: "1929-04-24", said: "I believe in Spinoza's God, who reveals himself in the harmony of all that exists, not in a God who concerns himself with the fates and actions of human beings.", facts: &[
        fact!("spinoza", "Spinoza's God", "Albert believes in 'Spinoza's God', a pantheistic view of God as synonymous with nature and the laws of physics, not a personal deity.", "philosophy", ["spinoza", "god"], [("Spinoza", "person")]),
    ]},
    Message { date: "1929-10-26", said: "Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.", facts: &[
        fact!("imagination", "Imagination over knowledge", "Albert believes imagination is more important than knowledge, because knowledge is limited and imagination encircles the world.", "philosophy", ["imagination", "quote"], []),
    ]},
    Message { date: "1930-07-04", said: "I love sailing and have owned several sailboats. I sail on the lakes near Berlin, though I never learned to swim.", facts: &[
        fact!("sailing", "Loves sailing, never learned to swim", "Albert loves sailing, has owned several sailboats and sails on the lakes near Berlin, despite never having learned to swim.", "hobbies", ["sailing"], [("Berlin", "place")]),
    ]},
    Message { date: "1930-09-01", said: "I often play in string quartets. Music helps me think about physics problems.", facts: &[
        fact!("quartets", "Plays in string quartets", "Albert often performs in string quartets, and music helps him think about physics problems.", "hobbies", ["violin", "music"], []),
    ]},
    Message { date: "1930-11-09", said: "Science without religion is lame, religion without science is blind. When I contemplate the harmony of natural law I experience what I can only call a cosmic religious feeling.", facts: &[
        fact!("sci_rel", "Science and religion need each other", "Albert believes science without religion is lame, and religion without science is blind.", "philosophy", ["religion", "science"], []),
        fact!("cosmic", "A cosmic religious feeling", "Albert experiences a 'cosmic religious feeling' when contemplating the harmony of natural law.", "philosophy", ["religion", "awe"], []),
    ]},
    Message { date: "1931-02-04", said: "Edwin Hubble has shown that the universe is expanding. The cosmological constant was my biggest blunder.", facts: &[
        fact!("blunder", "The cosmological constant, his biggest blunder", "After Edwin Hubble's discovery that the universe is expanding, Albert called the cosmological constant his 'biggest blunder'.", "science", ["cosmological constant", "hubble"], [("Edwin Hubble", "person"), ("cosmological constant", "thing")]),
    ]},
    Message { date: "1933-03-30", said: "I will not return to Germany now that Hitler has come to power. As a Jew I am not safe there. My property has been confiscated and my German citizenship revoked.", facts: &[
        fact!("escape", "Fled Nazi Germany, 1933", "Albert, who is Jewish, fled Nazi Germany in 1933 when Hitler came to power; his property was confiscated and his German citizenship revoked.", "politics", ["nazi germany", "exile"], [("Hitler", "person"), ("Germany", "place")]),
    ]},
    Message { date: "1933-05-01", said: "My son Eduard struggles with schizophrenia. He is in a clinic in Zurich.", facts: &[
        fact!("eduard", "Son Eduard in a clinic in Zurich", "Albert's son Eduard struggles with schizophrenia and is in a clinic in Zurich.", "family", ["children", "eduard"], [("Eduard", "person"), ("Zurich", "place")]),
    ]},
    Message { date: "1933-06-10", said: "Everything should be made as simple as possible, but not simpler.", facts: &[
        fact!("simplicity", "As simple as possible, but not simpler", "Albert believes everything should be made as simple as possible, but not simpler.", "philosophy", ["simplicity", "quote"], []),
    ]},
    Message { date: "1933-10-17", said: "I have taken a post at the Institute for Advanced Study in Princeton, New Jersey.", facts: &[
        fact!("princeton", "Institute for Advanced Study, Princeton", "Albert works at the Institute for Advanced Study in Princeton, New Jersey.", "career", ["princeton"], [("Institute for Advanced Study", "org"), ("Princeton", "place"), ("New Jersey", "place")]),
    ]},
    Message { date: "1933-10-20", said: "I don't wear socks and I let my hair grow wild. I don't want to waste time on mundane matters.", facts: &[
        fact!("appearance", "No socks, wild hair", "Albert avoids wearing socks and lets his hair grow wild, because he doesn't want to waste time on mundane matters.", "personal", ["appearance"], []),
    ]},
    Message { date: "1935-05-15", said: "Podolsky, Rosen and I have published a paper arguing that quantum mechanics is incomplete. What it calls entanglement I call spooky action at a distance; hidden variables must explain these correlations.", facts: &[
        fact!("epr", "The EPR paradox paper, 1935", "In 1935 Albert published the EPR paradox paper with Podolsky and Rosen, arguing that quantum mechanics is incomplete.", "science", ["epr", "quantum mechanics"], [("Podolsky", "person"), ("Rosen", "person"), ("quantum mechanics", "thing")]),
        fact!("spooky", "'Spooky action at a distance'", "Albert calls quantum entanglement 'spooky action at a distance' and believes hidden variables must explain quantum correlations.", "philosophy", ["entanglement"], [("entanglement", "thing")]),
    ]},
    Message { date: "1936-10-01", said: "The most incomprehensible thing about the universe is that it is comprehensible.", facts: &[
        fact!("mystery", "A comprehensible universe", "Albert thinks the most incomprehensible thing about the universe is that it is comprehensible.", "philosophy", ["universe", "quote"], []),
    ]},
    Message { date: "1936-10-05", said: "Education is what remains after one has forgotten what one has learned in school.", facts: &[
        fact!("education", "What education is", "Albert believes education is what remains after one has forgotten what one has learned in school.", "philosophy", ["education", "quote"], []),
    ]},
    Message { date: "1936-12-20", said: "Elsa died today.", facts: &[
        fact!("elsa_died", "Elsa died, 1936", "Albert's wife Elsa Löwenthal died in 1936.", "family", ["elsa", "loss"], [("Elsa Löwenthal", "person")]),
    ]},
    Message { date: "1939-08-02", said: "I have signed a letter to President Roosevelt warning that Germany might develop atomic weapons and urging America to begin its own research.", facts: &[
        fact!("letter", "The letter to President Roosevelt, 1939", "In 1939 Albert signed a letter to President Roosevelt warning that Nazi Germany might develop atomic weapons, which helped start the Manhattan Project.", "politics", ["roosevelt", "atomic bomb"], [("President Roosevelt", "person"), ("Manhattan Project", "org"), ("Germany", "place")]),
    ]},
    Message { date: "1940-10-01", said: "I became an American citizen today, though I am keeping my Swiss citizenship. I will never return to Germany.", facts: &[
        fact!("uscitizen", "Became a US citizen, 1940", "Albert became a US citizen in 1940, kept his citizenship of Switzerland, and has never returned to Germany since 1933.", "biography", ["citizenship", "america"], [("Germany", "place"), ("Switzerland", "place")]),
    ]},
    Message { date: "1945-08-09", said: "After Hiroshima I deeply regret my role in the atomic bomb's development. Had I known, I would have become a watchmaker.", facts: &[
        fact!("regret", "Regret over the atomic bomb", "After Hiroshima, Albert deeply regrets his role in the atomic bomb's development and says that had he known, he would have become a watchmaker.", "politics", ["atomic bomb", "regret"], [("Hiroshima", "place"), ("atomic bomb", "thing")]),
    ]},
    Message { date: "1946-05-03", said: "I speak out against racism in America. I have called racism a disease of white people, and I support the civil rights movement.", facts: &[
        fact!("civil", "Speaks out against racism in America", "Albert speaks out against racism in America, has called racism 'a disease of white people', and supports the civil rights movement.", "politics", ["civil rights", "racism"], [("America", "place")]),
    ]},
    Message { date: "1947-03-10", said: "I signed the letter to Roosevelt because I feared Germany would build the bomb first. It was the one exception I have made to my pacifism, and I remain a pacifist.", facts: &[
        fact!("why_signed", "Why a pacifist signed the Roosevelt letter", "Albert signed the Roosevelt letter only because he feared Germany would build the bomb first, the one exception to his pacifism, which he still holds.", "politics", ["pacifism", "roosevelt"], [("Roosevelt", "person"), ("Germany", "place")]),
    ]},
    Message { date: "1947-09-01", said: "My son Hans Albert has become a professor of hydraulic engineering at Berkeley.", facts: &[
        fact!("hans", "Son Hans Albert, a professor at Berkeley", "Albert's son Hans Albert became a professor of hydraulic engineering at Berkeley.", "family", ["children", "hans albert"], [("Hans Albert", "person"), ("Berkeley", "org")]),
    ]},
    Message { date: "1949-03-14", said: "I have spent nearly 30 years trying to develop a unified field theory combining gravity and electromagnetism, without success so far.", facts: &[
        fact!("unified", "The quest for a unified field theory", "Albert has spent nearly 30 years trying to develop a unified field theory combining gravity and electromagnetism, without success so far.", "science", ["unified field theory"], [("unified field theory", "thing")]),
    ]},
    Message { date: "1949-05-01", said: "I have published 'Why Socialism?', arguing for a planned economy to overcome the economic anarchy of capitalism.", facts: &[
        fact!("socialism", "Published 'Why Socialism?', 1949", "In 1949 Albert published 'Why Socialism?', arguing for a planned economy to overcome the 'economic anarchy' of capitalism.", "politics", ["socialism"], [("Why Socialism?", "thing")]),
    ]},
    Message { date: "1950-03-01", said: "J. Edgar Hoover's FBI keeps a file on me and monitors me as a potential subversive because of my political views.", facts: &[
        fact!("fbi", "Under FBI surveillance", "J. Edgar Hoover's FBI keeps a file on Albert, monitoring him as a potential subversive because of his political views.", "politics", ["fbi", "surveillance"], [("J. Edgar Hoover", "person"), ("FBI", "org")]),
    ]},
    Message { date: "1952-11-17", said: "I have been offered the presidency of Israel. I have declined; I am not suited to it.", facts: &[
        fact!("israel", "Declined the presidency of Israel, 1952", "In 1952 Albert was offered the presidency of Israel and declined.", "politics", ["israel", "presidency"], [("Israel", "place")]),
    ]},
];

/// Links the way the app writes them: the newer memory is the source. "updates" marks
/// the target as replaced; "resolves" is the user's own explanation, so it is "by the user".
pub static LINKS: &[Link] = &[
    // updates: a later state of the same thing
    Link { source: "berlin", target: "patent", relation: "updates", confidence: 0.95, note: "He left the patent office for academic posts; Berlin is the current one" },
    Link { source: "princeton", target: "berlin", relation: "updates", confidence: 0.95, note: "The Berlin professorship ended when he left Germany; Princeton is his post now" },
    Link { source: "blunder", target: "cosmological", relation: "updates", confidence: 0.95, note: "He withdrew the cosmological constant after Hubble's discovery" },
    Link { source: "divorce", target: "mileva", relation: "updates", confidence: 0.95, note: "The marriage to Mileva ended in divorce" },
    Link { source: "elsa_died", target: "elsa", relation: "updates", confidence: 0.9, note: "Elsa's death ended the marriage recorded in 1919" },
    // contradicts
    Link { source: "letter", target: "pacifism", relation: "contradicts", confidence: 0.9, note: "A committed pacifist signed the letter that launched the Manhattan Project" },
    Link { source: "determinism", target: "bose", relation: "contradicts", confidence: 0.7, note: "Denies that nature is governed by chance, yet built a statistics of chance into physics" },
    // resolves: the user's explanation of how both can be true
    Link { source: "why_signed", target: "letter", relation: "resolves", confidence: 0.95, note: "The user's explanation of how both can be true" },
    Link { source: "why_signed", target: "pacifism", relation: "resolves", confidence: 0.95, note: "The user's explanation of how both can be true" },
    // related
    Link { source: "mother", target: "parents", relation: "related", confidence: 0.9, note: "Pauline is the mother named among his parents" },
    Link { source: "swiss", target: "eth", relation: "related", confidence: 0.7, note: "Swiss education, then Swiss citizenship" },
    Link { source: "patent", target: "eth", relation: "related", confidence: 0.6, note: "The job that followed his studies" },
    Link { source: "olympia", target: "patent", relation: "related", confidence: 0.8, note: "Both from his Bern years" },
    Link { source: "patent_light", target: "patent", relation: "related", confidence: 0.9, note: "His patent work and what it did to his thinking" },
    Link { source: "miracle", target: "photoelectric", relation: "related", confidence: 0.9, note: "One of the four 1905 papers" },
    Link { source: "miracle", target: "brownian", relation: "related", confidence: 0.9, note: "One of the four 1905 papers" },
    Link { source: "miracle", target: "sr", relation: "related", confidence: 0.9, note: "One of the four 1905 papers" },
    Link { source: "miracle", target: "emc2", relation: "related", confidence: 0.9, note: "One of the four 1905 papers" },
    Link { source: "emc2", target: "sr", relation: "related", confidence: 0.9, note: "Mass-energy equivalence follows from special relativity" },
    Link { source: "equivalence", target: "sr", relation: "related", confidence: 0.8, note: "The step from special to general relativity" },
    Link { source: "lensing", target: "equivalence", relation: "related", confidence: 0.8, note: "Light bending follows from the equivalence principle" },
    Link { source: "spacetime", target: "equivalence", relation: "related", confidence: 0.8, note: "Curved spacetime is where the equivalence principle led" },
    Link { source: "gr", target: "equivalence", relation: "related", confidence: 0.9, note: "The equivalence principle was the seed of general relativity" },
    Link { source: "gr", target: "spacetime", relation: "related", confidence: 0.9, note: "Curved spacetime is the heart of general relativity" },
    Link { source: "gr", target: "lensing", relation: "related", confidence: 0.8, note: "Light bending is a prediction of general relativity" },
    Link { source: "gr", target: "sr", relation: "related", confidence: 0.8, note: "General relativity extends special relativity to gravity" },
    Link { source: "blackholes", target: "gr", relation: "related", confidence: 0.9, note: "Black holes fall out of the field equations" },
    Link { source: "gravwaves", target: "gr", relation: "related", confidence: 0.9, note: "Gravitational waves are a prediction of general relativity" },
    Link { source: "cosmological", target: "gr", relation: "related", confidence: 0.85, note: "The constant was added to the general relativity equations" },
    Link { source: "eclipse", target: "lensing", relation: "related", confidence: 0.95, note: "The eclipse confirmed the light-bending prediction" },
    Link { source: "eclipse", target: "gr", relation: "related", confidence: 0.85, note: "The confirmation that made general relativity famous" },
    Link { source: "nobel", target: "photoelectric", relation: "related", confidence: 0.95, note: "The prize was for the photoelectric effect paper" },
    Link { source: "bose", target: "photoelectric", relation: "related", confidence: 0.7, note: "Both are statistics of light quanta" },
    Link { source: "qm", target: "photoelectric", relation: "related", confidence: 0.8, note: "The photoelectric paper is how he helped found quantum mechanics" },
    Link { source: "bohr", target: "qm", relation: "related", confidence: 0.9, note: "The debates are where his skepticism played out" },
    Link { source: "spinoza", target: "determinism", relation: "related", confidence: 0.85, note: "God as the laws of nature is a deterministic universe" },
    Link { source: "sci_rel", target: "spinoza", relation: "related", confidence: 0.8, note: "The religion he means is Spinoza's" },
    Link { source: "cosmic", target: "spinoza", relation: "related", confidence: 0.85, note: "The feeling that goes with Spinoza's God" },
    Link { source: "mystery", target: "cosmic", relation: "related", confidence: 0.8, note: "Awe at a comprehensible universe" },
    Link { source: "education", target: "imagination", relation: "related", confidence: 0.7, note: "Both on what matters more than schooling" },
    Link { source: "unified", target: "gr", relation: "related", confidence: 0.8, note: "Unification builds on general relativity" },
    Link { source: "unified", target: "simplicity", relation: "related", confidence: 0.7, note: "One theory for everything is the simplicity he believes in" },
    Link { source: "epr", target: "qm", relation: "related", confidence: 0.9, note: "The EPR paper is his case that quantum mechanics is incomplete" },
    Link { source: "epr", target: "bohr", relation: "related", confidence: 0.8, note: "The paper continued the argument with Bohr" },
    Link { source: "wwi", target: "pacifism", relation: "related", confidence: 0.95, note: "The first public act of his pacifism" },
    Link { source: "escape", target: "berlin", relation: "related", confidence: 0.9, note: "He left Berlin when Hitler came to power" },
    Link { source: "princeton", target: "escape", relation: "related", confidence: 0.9, note: "Princeton gave him a post after he fled Germany" },
    Link { source: "uscitizen", target: "swiss", relation: "related", confidence: 0.8, note: "He kept the Swiss citizenship" },
    Link { source: "uscitizen", target: "escape", relation: "related", confidence: 0.85, note: "Exile led to American citizenship" },
    Link { source: "uscitizen", target: "princeton", relation: "related", confidence: 0.7, note: "Citizenship followed the Princeton post" },
    Link { source: "letter", target: "escape", relation: "related", confidence: 0.8, note: "Fear of the Germany he fled building the bomb" },
    Link { source: "regret", target: "letter", relation: "related", confidence: 0.95, note: "The letter he regrets" },
    Link { source: "why_signed", target: "regret", relation: "related", confidence: 0.8, note: "The regret and the reason" },
    Link { source: "civil", target: "pacifism", relation: "related", confidence: 0.7, note: "Both oppose organised violence and oppression" },
    Link { source: "socialism", target: "civil", relation: "related", confidence: 0.7, note: "Economic and racial oppression" },
    Link { source: "fbi", target: "socialism", relation: "related", confidence: 0.85, note: "The FBI file follows his socialist writing" },
    Link { source: "fbi", target: "civil", relation: "related", confidence: 0.7, note: "His civil rights activism is in the file" },
    Link { source: "fbi", target: "pacifism", relation: "related", confidence: 0.6, note: "The political views the FBI watches" },
    Link { source: "israel", target: "zionism", relation: "related", confidence: 0.95, note: "The offer came because of his support for Zionism" },
    Link { source: "quartets", target: "violin", relation: "related", confidence: 0.9, note: "The violin he learned as a child" },
    Link { source: "sailing", target: "berlin", relation: "related", confidence: 0.6, note: "The lakes near Berlin" },
    Link { source: "appearance", target: "pipe", relation: "related", confidence: 0.6, note: "Indifference to convention" },
    Link { source: "elsa", target: "divorce", relation: "related", confidence: 0.8, note: "Remarried the year of the divorce" },
    Link { source: "children", target: "mileva", relation: "related", confidence: 0.9, note: "The children of the marriage to Mileva" },
    Link { source: "divorce", target: "children", relation: "related", confidence: 0.7, note: "The children stayed with Mileva" },
    Link { source: "eduard", target: "children", relation: "related", confidence: 0.9, note: "One of the three children" },
    Link { source: "hans", target: "children", relation: "related", confidence: 0.9, note: "One of the three children" },
    Link { source: "hans", target: "eduard", relation: "related", confidence: 0.7, note: "His two sons" },
];

/// The name-finder shape the memories table keeps (`entities_detected`), so the map and
/// recall treat demo memories like any other, and so the names travel with sync and backup.
fn entities_json(fact: &Fact) -> String {
    let items: Vec<serde_json::Value> = fact.entities.iter().map(|(name, kind)| {
        let label = match *kind { "person" => "PER", "place" => "LOC", "org" => "ORG", _ => "MISC" };
        let start = fact.fact.find(name).unwrap_or(0);
        serde_json::json!({ "word": name, "label": label, "score": 0.95, "start": start, "end": start + name.len() })
    }).collect();
    serde_json::Value::Array(items).to_string()
}

fn timestamp(date: &str) -> String {
    format!("{date}T12:00:00+00:00")
}

/// Replace any earlier copy of the demo with this one. Embeddings come from the local
/// model; when it is not available the memories are stored without one and a warning
/// is logged (recall by meaning then misses them until the next re-embed).
pub async fn load(pool: &SqlitePool, user_id: &str) -> Result<(usize, usize), String> {
    sqlx::query("DELETE FROM memories WHERE session_id IN (?, 'einstein-demo-small-session') OR (namespace = 'einstein' AND source_type = 'demo_data')")
        .bind(SESSION_ID).execute(pool).await.map_err(|e| format!("Could not clear the old demo: {e}"))?;

    let mut ids: std::collections::HashMap<&'static str, i32> = std::collections::HashMap::new();
    let mut texts: Vec<(i32, String)> = Vec::new();
    for m in MESSAGES {
        let when = timestamp(m.date);
        for f in m.facts {
            let tags = serde_json::to_string(f.tags).unwrap_or_else(|_| "[]".into());
            let id: i32 = sqlx::query_scalar(
                "INSERT INTO memories (title, content, original_text, namespace, user_id, session_id, source_type, is_syncable, is_shareable, entities_detected, tags, created_at, updated_at, event_date)
                 VALUES (?, ?, ?, ?, ?, ?, 'demo_data', 1, 0, ?, ?, ?, ?, ?) RETURNING id")
                .bind(f.title).bind(f.fact).bind(m.said).bind(f.namespace).bind(user_id).bind(SESSION_ID)
                .bind(entities_json(f)).bind(&tags).bind(&when).bind(&when).bind(&when)
                .fetch_one(pool).await.map_err(|e| format!("Could not store '{}': {e}", f.title))?;
            ids.insert(f.key, id);
            texts.push((id, crate::memory::embedding_text(Some(f.title), f.fact)));
            let entities: Vec<crate::memory_extras::EntityOut> = f.entities.iter()
                .map(|(name, kind)| crate::memory_extras::EntityOut { name: name.to_string(), kind: kind.to_string() }).collect();
            if let Err(e) = crate::memory::insert_memory_entities(pool, id, &entities).await {
                println!("[Einstein] ⚠️ Could not record the named things of '{}': {e}", f.title);
            }
        }
    }

    let batch: Vec<String> = texts.iter().map(|(_, t)| t.clone()).collect();
    match tokio::task::spawn_blocking(move || crate::llm::local_embeddings::generate_local_embeddings_batch(batch, Some(32))).await {
        Ok(Ok(embs)) => {
            for ((id, _), e) in texts.iter().zip(embs) {
                let blob: Vec<u8> = e.iter().flat_map(|f| f.to_le_bytes()).collect();
                sqlx::query("UPDATE memories SET embedding = ? WHERE id = ?").bind(blob).bind(id)
                    .execute(pool).await.map_err(|e| format!("Could not store an embedding: {e}"))?;
            }
        }
        Ok(Err(e)) => println!("[Einstein] ⚠️ No embeddings for the demo (local model): {e}"),
        Err(e) => println!("[Einstein] ⚠️ No embeddings for the demo (task): {e}"),
    }

    let mut links = 0usize;
    for l in LINKS {
        let (Some(&s), Some(&t)) = (ids.get(l.source), ids.get(l.target)) else { continue };
        let by = if l.relation == "resolves" { "user" } else { "system" };
        match crate::memory::create_memory_link(pool, s, t, l.relation, l.confidence, Some(l.note), by).await {
            Ok(_) => links += 1,
            Err(e) => println!("[Einstein] ⚠️ Could not link {} -> {}: {e}", l.source, l.target),
        }
        if l.relation == "updates" {
            if let Err(e) = crate::memory::set_temporal_status(pool, t, Some("superseded")).await {
                println!("[Einstein] ⚠️ Could not mark {} as replaced: {e}", l.target);
            }
        }
    }
    Ok((ids.len(), links))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn facts() -> Vec<&'static Fact> { MESSAGES.iter().flat_map(|m| m.facts.iter()).collect() }

    #[test]
    fn every_fact_is_written_the_way_the_extractor_writes_one() {
        let mut keys = HashSet::new();
        let mut problems: Vec<String> = Vec::new();
        let mut check = |ok: bool, what: String| if !ok { problems.push(what) };
        for f in facts() {
            check(keys.insert(f.key), format!("duplicate key {}", f.key));
            check(f.title.chars().count() <= 50, format!("title too long: {}", f.title));
            check(f.fact.contains(NAME), format!("fact does not name the user: {}", f.fact));
            check(f.fact.ends_with('.'), format!("fact is not a sentence: {}", f.fact));
            check(crate::memory_extras::CANONICAL_NAMESPACES.contains(&f.namespace), format!("unknown namespace {} on {}", f.namespace, f.key));
            check(!f.tags.is_empty() && f.tags.len() <= 3, format!("{} needs 1 to 3 tags", f.key));
            for (name, kind) in f.entities {
                check(["person", "place", "org", "thing"].contains(kind), format!("{}: kind {kind}", f.key));
                check(f.fact.contains(name), format!("{}: named thing '{name}' is not in the fact", f.key));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
        for m in MESSAGES {
            assert!(chrono::NaiveDate::parse_from_str(m.date, "%Y-%m-%d").is_ok(), "bad date {}", m.date);
            assert!(!m.facts.is_empty());
        }
    }

    #[test]
    fn every_link_joins_two_known_facts_with_an_allowed_type() {
        let keys: HashSet<&str> = facts().iter().map(|f| f.key).collect();
        let mut seen = HashSet::new();
        for l in LINKS {
            assert!(keys.contains(l.source), "unknown source {}", l.source);
            assert!(keys.contains(l.target), "unknown target {}", l.target);
            assert_ne!(l.source, l.target);
            assert!(crate::memory::RELATION_TYPES.contains(&l.relation), "{} is not a relation type", l.relation);
            assert!((0.0..=1.0).contains(&l.confidence));
            assert!(seen.insert((l.source, l.target, l.relation)), "duplicate link {} -> {}", l.source, l.target);
            assert!(!seen.contains(&(l.target, l.source, l.relation)), "link both ways {} <-> {}", l.source, l.target);
        }
        let n = |t: &str| LINKS.iter().filter(|l| l.relation == t).count();
        assert!(n("updates") >= 3 && n("contradicts") >= 1 && n("resolves") >= 2 && n("related") >= 30, "the demo should show every kind of link");
    }

    #[test]
    fn a_replaced_memory_is_older_than_its_replacement() {
        let date_of = |key: &str| MESSAGES.iter().find(|m| m.facts.iter().any(|f| f.key == key)).map(|m| m.date).unwrap();
        for l in LINKS.iter().filter(|l| l.relation == "updates") {
            assert!(date_of(l.source) > date_of(l.target), "{} should be newer than {}", l.source, l.target);
        }
    }

    #[test]
    fn named_things_keep_the_name_finder_shape() {
        let f = facts().into_iter().find(|f| f.key == "parents").unwrap();
        let v: serde_json::Value = serde_json::from_str(&entities_json(f)).unwrap();
        assert_eq!(v[0]["word"], "Hermann Einstein");
        assert_eq!(v[0]["label"], "PER");
        assert_eq!(v[1]["label"], "PER");
        let names = crate::commands::memory::named_things_from_ner(&entities_json(f));
        assert_eq!(names, vec![("Hermann Einstein".to_string(), "person".to_string()), ("Pauline Koch".to_string(), "person".to_string())]);
    }
}
