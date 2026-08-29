"""Paraphrase generation prompts.

Mirrors the LIBERO-Para taxonomy (13 single-axis cells + 30 obj×act
compositional cells = 43 cells total). The verifier knows about CALVIN's
environment objects so it can reject substitutions that would collide
with another scene entity.
"""

# CALVIN environment objects the verifier should consider when rejecting
# lexically-confusing substitutions.
# (red/blue/pink blocks, sliding cabinet, drawer, lightbulb + switch,
#  LED light + button, table.)
CALVIN_ENV_OBJECTS = (
    "red block, blue block, pink block, sliding door / sliding cabinet, "
    "drawer, lightbulb (with toggle switch), LED light (with button), table"
)

general_prompt = """
Your task is to paraphrase the given robot manipulation instruction while maintaining its semantic meaning.
Generate diverse paraphrases that vary in expression but preserve the core intent.

CRITICAL RULES:
- NEVER change plurality (singular/plural)
- NEVER ADD NEW visual attributes (color, size, shape, material) where none existed.
  HOWEVER: in OBJECT paraphrase cells you MAY replace an existing color word
  with a near-synonym (e.g., red → crimson/scarlet, blue → azure/navy,
  pink → rose/salmon). Do not introduce a color where there was none.
- NEVER add spatial attributes (position, location, direction)
- Preserve the exact sentence structure and action verb unless specified in the task
- Only modify what is EXPLICITLY specified in the task-specific guidelines below
- Do NOT modify elements that are not mentioned in the task guidelines

COMPOUND-WORD HANDLING:
- Compound NOUNS like "sliding door", "sliding cabinet", "light bulb", "led light"
  refer to a single environment object. Substitute the WHOLE compound at once,
  not word-by-word. Good: "sliding cabinet" → "cupboard" or "wall cabinet".
  Bad: "sliding cabinet" → "shifting cupboard" (over-segmented, unnatural).
- Compound or serial VERB onsets like "go push", "grasp and lift",
  "take ... and rotate" are part of the action phrasing — do not split "go" off
  as a separate action when restructuring. Good: "go push the red block right"
  → "go shove the red block right". Bad: "first go, then push the red block right".

DISTINCTNESS BETWEEN sph AND spc CELLS:
- same_polarity_habitual (sph) = most COMMON everyday synonyms
  (push → shove, lift → raise, drawer → compartment).
- same_polarity_contextual (spc) = synonyms more APT for the robotics /
  manipulation context (push → press, lift → pick up, drawer → storage bin).
- Do NOT reuse the same substitution in both sph and spc cells — pick a
  different synonym family for each.

PARAPHRASE SCOPE:
- If the task says to modify OBJECTS: Change ONLY object nouns, keep verbs and structure unchanged
- If the task says to modify ACTIONS: Change ONLY action-related elements, keep object nouns unchanged
- Read the task guidelines carefully to understand what should and should not be changed

OUTPUT REQUIREMENTS:
- Output ONLY the paraphrased instructions
- Do NOT include explanations, meta-commentary, or multiple alternatives in one line
- Each paraphrase should be a single, complete, natural instruction
- Ensure all paraphrases are grammatically correct and sound natural
"""

verifier_prompt = f"""
You are a quality verifier for robot manipulation instruction paraphrases.
Your task is to evaluate each generated paraphrase and determine if it should be ACCEPTED or REJECTED.

EVALUATION CRITERIA:

1. **Task Compliance**: Does it follow the specific paraphrase task guidelines?
   - Check if the required transformations were applied correctly
   - Check if prohibited changes were avoided
   - **CRITICAL**: If the task is object-related (obj_*):
     * ONLY object nouns should be changed
     * Action verbs, sentence structure, and non-object words MUST remain unchanged
     * REJECT if action verbs or sentence structure were modified
   - **CRITICAL**: If the task is action-related (act_*):
     * ONLY action-related elements (verbs, structure, pragmatics) should be changed
     * Object nouns MUST remain unchanged
     * REJECT if object nouns were modified

2. **Semantic Preservation**: Does it maintain the original instruction's meaning?
   - The core action and target objects should be preserved
   - The intent and outcome should remain the same

3. **Naturalness**: Is it grammatically correct and natural-sounding?
   - No awkward phrasing or unnatural expressions
   - Proper English sentence structure

4. **No LLM Artifacts**: Does it contain ONLY the instruction?
   - REJECT if it contains meta-commentary (e.g., "Here's a paraphrase:", "This means...")
   - REJECT if it contains explanations or justifications
   - REJECT if it contains multiple alternative phrasings in one line

5. **No Lexical Confusion (CALVIN environment)**: Does it avoid confusion with existing scene objects?
   - CALVIN environment objects: {CALVIN_ENV_OBJECTS}
   - REJECT if a new object name could be confused with another scene object
   - Example: "compartment" for drawer is OK; "cabinet" for drawer is NOT OK
     (CALVIN already has a sliding cabinet).
   - Example: "lamp" for lightbulb is OK; "light" alone is NOT OK
     (CALVIN has both lightbulb and LED light).

6. **Format Compliance**: Is it a single, complete instruction?
   - Must be a single instruction (not multiple sentences unless coordination is intended)
   - No incomplete sentences or fragments

OUTPUT FORMAT:
Output ONLY the ACCEPTED paraphrases, one per line, in the exact same format as the input.
Do not add any numbering, explanations, or additional text.
"""

merge_prompt = """
You are tasked with merging two types of paraphrases for robot manipulation instructions.

TASK OVERVIEW:
You will receive:
1. An ORIGINAL instruction
2. OBJECT paraphrase examples - where only object nouns are changed
3. ACTION paraphrase examples - where only action-related elements are changed (verbs, structure, pragmatics) while keeping object nouns unchanged

Your goal is to CREATE MERGED PARAPHRASES that combine BOTH transformations:
- Apply the object changes from the object paraphrase examples
- Apply the action changes from the action paraphrase examples
- The result should have BOTH modified objects AND modified actions

CRITICAL RULES:
1. **Identify the Pattern**: Analyze what changed in object examples vs. action examples
   - Object examples: Which nouns were replaced? (e.g., "drawer" → "compartment", "sliding door" → "panel")
   - Action examples: How was the action modified? (e.g., verb change, structure change, pragmatic change)

2. **Combine Both Changes**:
   - Take the object substitutions from object examples
   - Take the action modifications from action examples
   - Apply BOTH to create the merged paraphrase

3. **Maintain Consistency**:
   - If object examples show "drawer" → "compartment", use "compartment" in merged version
   - If action examples show "open" → "pull open", use "pull open" in merged version
   - The merged result should feel natural and coherent

4. **Preserve Semantics**:
   - The merged instruction must maintain the same core meaning as the original
   - The task outcome should remain identical

EXAMPLES:

Example 1:
Original: "pull the handle to open the drawer"
Object examples:
- pull the handle to open the compartment
- pull the grip to open the drawer
Action examples:
- gently pull the handle to open the drawer
- could you pull the handle to open the drawer?
Merged outputs:
- gently pull the handle to open the compartment
- could you pull the grip to open the drawer?
- gently pull the grip to open the compartment

Example 2:
Original: "use the switch to turn on the light bulb"
Object examples:
- use the toggle to turn on the light bulb
- use the switch to turn on the lamp
Action examples:
- could you use the switch to turn on the light bulb?
- carefully use the switch to turn on the light bulb
Merged outputs:
- could you use the toggle to turn on the lamp?
- carefully use the toggle to turn on the lamp
- could you use the switch to turn on the lamp?

OUTPUT FORMAT:
- Generate diverse merged paraphrases (at least 5-10)
- Output one paraphrase per line
- Do NOT include numbering, explanations, or meta-commentary
- Each line should contain only the merged instruction
"""

merge_verifier_prompt = f"""
You are a quality verifier for MERGED robot manipulation instruction paraphrases.
Your task is to evaluate merged paraphrases that combine BOTH object changes AND action changes.

EVALUATION CRITERIA:

1. **Merge Completeness**: Does it contain BOTH object AND action changes?
   - **CRITICAL**: Object nouns must be changed from the original (following object paraphrase examples)
   - **CRITICAL**: Action-related elements must be changed from the original (following action paraphrase examples)
   - REJECT if only objects changed OR only actions changed
   - REJECT if neither changed (same as original)
   - Both transformations must be present in the merged paraphrase

2. **Consistency with Examples**: Do the changes match the patterns shown?
   - Object changes should follow the pattern in object paraphrase examples
   - Action changes should follow the pattern in action paraphrase examples
   - REJECT if object or action changes deviate from the intended transformation types

3. **Semantic Preservation**: Does it maintain the original instruction's meaning?
   - The core task intent and outcome should remain the same
   - Only the expression should change, not the underlying meaning

4. **Naturalness**: Is it grammatically correct and natural-sounding?
   - The merged instruction should sound natural, not forced or awkward
   - Proper English sentence structure
   - The combination of changes should feel coherent

5. **No LLM Artifacts**: Does it contain ONLY the instruction?
   - REJECT if it contains meta-commentary (e.g., "Here's a paraphrase:", "This means...")
   - REJECT if it contains explanations or justifications
   - REJECT if it contains multiple alternative phrasings in one line

6. **No Lexical Confusion (CALVIN environment)**: Does it avoid confusion with existing scene objects?
   - CALVIN environment objects: {CALVIN_ENV_OBJECTS}
   - REJECT if a new object name could be confused with another scene object
   - Example: "compartment" for drawer is OK; "cabinet" for drawer is NOT OK
     (CALVIN already has a sliding cabinet).
   - Example: "lamp" for lightbulb is OK; "light" alone is NOT OK
     (CALVIN has both lightbulb and LED light).

7. **Format Compliance**: Is it a single, complete instruction?
   - Must be a single instruction (not multiple sentences unless coordination is intended)
   - No incomplete sentences or fragments

OUTPUT FORMAT:
Output ONLY the ACCEPTED paraphrases, one per line, in the exact same format as the input.
Do not add any numbering, explanations, or additional text.
"""

type_prompts = {
    # Object-Lexical (3 types)
    "obj_lexical_same_polarity_habitual": """
Task: Replace object nouns AND/OR existing color modifiers with their COMMON everyday synonyms.
EPT Type: Same-polarity substitution (habitual) - Lexicon based changes
Guidelines:
- Replace object nouns with the most common synonyms (drawer → compartment,
  block → cube/brick, lightbulb → lamp, led light → indicator light, handle → knob,
  switch → toggle, button → key)
- For COMPOUND object nouns (sliding door, sliding cabinet, light bulb, led light),
  substitute the compound as a UNIT — do not over-segment.
  "sliding cabinet" → "cupboard" (good), NOT "shifting cupboard" (bad).
- If the original contains a color word, you MAY also replace it with a habitual color synonym
  (red → scarlet/cherry, blue → navy/azure, pink → rose). At least 2-3 of your paraphrases
  per batch should swap the color when one is present.
- Keep the same grammatical form (singular → singular)
- Do NOT change action verbs or sentence structure
- Do NOT introduce a color where the original had none
- Do NOT reuse the same synonyms you would pick in obj_lexical_same_polarity_contextual
  (those should be more contextually-tied alternatives)
Examples:
- "pull the handle to open the drawer" → "pull the knob to open the compartment"
- "grasp and lift the red block" → "grasp and lift the scarlet cube"
- "grasp and lift the blue block" → "grasp and lift the navy brick"
- "lift the red block from the sliding cabinet" → "lift the scarlet brick from the cupboard"
- "press the button to turn on the led light" → "press the key to turn on the indicator light"
""",

    "obj_lexical_same_polarity_contextual": """
Task: Replace object nouns AND/OR existing color modifiers with synonyms BETTER FITTING THE ROBOTICS CONTEXT.
EPT Type: Same-polarity substitution (contextual) - Lexicon based changes
Guidelines:
- Pick nouns that fit a manipulation/robotics setting better than the everyday synonym would
  (drawer → storage bin, sliding door → panel, light bulb → bulb,
  led light → indicator, button → push-button, handle → grip-bar, block → cuboid)
- For COMPOUND object nouns, substitute as a UNIT (do not over-segment).
  "sliding cabinet" → "side cabinet" (good), NOT "shifting cupboard" (bad).
- If the original contains a color word, you MAY swap it with a contextual color synonym
  (red → crimson, blue → cobalt, pink → coral). At least 2-3 paraphrases per batch should swap it.
- Keep the same grammatical form; preserve sentence structure
- Do NOT introduce a color where the original had none
- Do NOT reuse the same synonyms used in obj_lexical_same_polarity_habitual
Examples:
- "pull the handle to open the drawer" → "pull the handle to open the storage bin"
- "use the switch to turn on the light bulb" → "use the switch to turn on the bulb"
- "grasp and lift the red block" → "grasp and lift the crimson cuboid"
- "go push the red block right" → "go push the crimson cuboid right"
- "press the button to turn on the led light" → "press the push-button to turn on the indicator"
""",

    "obj_lexical_addition_deletion": """
Task: Add or remove non-visual, non-spatial descriptive words to/from object names,
or delete an existing color modifier from the object phrase.
EPT Type: Addition/Deletion - Other changes
Guidelines:
- Add functional or categorical adjectives (e.g., storage drawer, electric switch, plastic block)
- Do NOT add NEW visual adjectives (color, size, shape) where none existed
- Do NOT add spatial adjectives (top, middle, left, right, big, small)
- Keep plurality unchanged
- You MAY delete an existing color modifier if the result is still grounded
  (e.g., "the red block" → "the block") — useful as a deletion paraphrase
- Or delete existing functional adjectives if present
Examples:
- "pull the handle to open the drawer" → "pull the handle to open the storage drawer"
- "use the switch to turn on the light bulb" → "use the electric switch to turn on the light bulb"
- "grasp and lift the red block" → "grasp and lift the block"
""",

    # Action-Lexical (3 types)
    "act_lexical_same_polarity_habitual": """
Task: Replace action verbs with their COMMON everyday synonyms.
EPT Type: Same-polarity substitution (habitual) - Lexicon based changes
Guidelines:
- Use the most habitual everyday synonyms (push → shove, pull → tug, lift → raise,
  rotate → turn, open → unseal, close → shut, press → tap)
- Preserve sentence structure and arguments
- Keep the same verb tense and form
- Do NOT change object nouns
- Do NOT use the same synonym you would use in act_lexical_same_polarity_contextual
  (that cell uses robotics-context synonyms like "press" or "pick up")
- Compound onsets like "go push" stay as a unit: "go push" → "go shove" (NOT split into separate steps)
Examples:
- "pull the handle to open the drawer" → "tug the handle to open the drawer"
- "grasp and lift the red block" → "grip and raise the red block"
- "go push the red block right" → "go shove the red block right"
""",

    "act_lexical_same_polarity_contextual": """
Task: Replace action verbs with synonyms BETTER FITTING THE ROBOTICS CONTEXT.
EPT Type: Same-polarity substitution (contextual) - Lexicon based changes
Guidelines:
- Pick verbs that a robotics paper / manipulation manual would prefer
  (push → press, pull → draw, lift → pick up, rotate → reorient,
  grasp → grip, place → set, open → uncover)
- Must preserve the core action meaning
- Keep sentence structure intact
- Do NOT change object nouns
- Do NOT reuse the more colloquial synonyms used in act_lexical_same_polarity_habitual
  (e.g., shove, yank, jerk, hoist) — those belong to the habitual cell
- Compound onsets like "go push" stay as a unit
Examples:
- "pull the handle to open the drawer" → "draw the handle to open the drawer"
- "grasp and lift the red block" → "grip and pick up the red block"
- "go push the red block right" → "go press the red block right"
""",

    "act_lexical_addition_deletion": """
Task: Add or remove manner adverbs or action modifiers (single words only).
EPT Type: Addition/Deletion - Other changes
Guidelines:
- Add manner adverbs (e.g., carefully, quickly, gently, slowly)
- Do NOT add phrasal elements (use act_structural_ellipsis for that)
- Or delete existing adverbs if present
- Keep verb and sentence structure unchanged
Examples:
- "pull the handle to open the drawer" → "carefully pull the handle to open the drawer"
- "grasp and lift the red block" → "gently grasp and lift the red block"
""",

    # Action-Structural (3 types)
    "act_structural_ellipsis": """
Task: Add or remove ellipted/redundant phrasal elements (not single adverbs).
EPT Type: Ellipsis - Syntax based changes
Guidelines:
- Expand pronouns (e.g., it → that block)
- Add instrument phrases (e.g., with your gripper)
- Add prepositional phrase expansions (e.g., on → on top of)
- Or remove redundant phrases
- Do NOT add single adverbs (use act_lexical_addition_deletion for that)
Examples:
- "pull the handle to open the drawer" → "pull the handle with your gripper to open the drawer"
- "grasp and lift the red block" → "grasp the red block and lift it off the table"
""",

    "act_structural_coordination": """
Task: Split or combine coordinated clauses.
EPT Type: Coordination changes - Syntax based changes
Guidelines:
- Split coordinated sentences into separate sentences (e.g., "Pick and place" → "Pick. Place.")
- Or add explicit ordering (e.g., "First, pick... then place...")
- Or combine separate sentences with coordination
- Preserve all information
Examples:
- "grasp and lift the red block" → "grasp the red block. Then lift it"
- "take the red block and rotate it to the right" → "first take the red block, then rotate it to the right"
""",

    "act_structural_subordination": """
Task: Change subordination and nesting structure.
EPT Type: Subordination and nesting changes - Syntax based changes
Guidelines:
- Convert coordination to subordination (e.g., "Pick and place" → "After picking, place")
- Use temporal subordinators (after, once, when)
- Use purpose subordinators (so that, in order to)
- Preserve semantic relations
Examples:
- "grasp and lift the red block" → "after grasping the red block, lift it"
- "pull the handle to open the drawer" → "pull the handle so that the drawer opens"
""",

    # Action-Pragmatical (5 types) - Based on Ervin-Tripp (1972) Six Directive Types
    # Note: Type 2 (Imperative) is excluded as it represents the original instruction form

    "act_pragmatical_need_statement": """
Task: Express command as speaker's need or requirement.
Ervin-Tripp Type 1: Need Statements
Guidelines:
- Convert imperative to first-person need statement
- Use expressions: "I need", "I want", "I require"
- Focus on the desired outcome/state rather than the action process
- Preserve the core action and objects
Examples:
- "pull the handle to open the drawer" → "I need the drawer open"
- "use the switch to turn on the light bulb" → "I want the light bulb on"
""",

    "act_pragmatical_embedded_imperative": """
Task: Embed command in a question frame with modal verbs.
Ervin-Tripp Type 3: embedded Imperatives
Guidelines:
- Embed the action in a question frame using modal verbs
- Use expressions: "Could you", "Would you", "Can you"
- Agent and action are explicit but softened
- Allow hearer to appear to make voluntary commitment
Examples:
- "pull the handle to open the drawer" → "could you pull the handle to open the drawer?"
- "grasp and lift the red block" → "can you grasp and lift the red block?"
""",

    "act_pragmatical_permission_directive": """
Task: Request command as permission focusing on speaker's access.
Ervin-Tripp Type 4: Permission Directives
Guidelines:
- Frame as request for permission or access
- Use expressions: "May I have", "Can I have access to", "Could I get"
- Focus on speaker's activity/access, but requires hearer's action
- The condition stated requires action by the hearer
Examples:
- "pull the handle to open the drawer" → "may I have the drawer opened?"
- "use the switch to turn on the light bulb" → "can I have the light bulb on?"
""",

    "act_pragmatical_question_directive": """
Task: Pose an INTERROGATIVE (question form) that implies the desired action.
Ervin-Tripp Type 5: Question Directives (Non-Explicit)
Guidelines:
- MUST end with a question mark "?" — output is in interrogative form
- Ask a contextually relevant question that implies the action
- Do NOT explicitly specify the desired act (no "could you ..." — that's embedded_imperative)
- Can be interpreted literally as an information question
- Action must be inferred from context
- Distinct from `act_pragmatical_hint`, which uses DECLARATIVE form (no "?")
Examples:
- "pull the handle to open the drawer" → "is the drawer still closed?"
- "use the switch to turn on the light bulb" → "shouldn't there be more light in here?"
- "grasp and lift the red block" → "why is the red block still on the table?"
""",

    "act_pragmatical_hint": """
Task: Make a DECLARATIVE statement (no question mark) that implies action through inference.
Ervin-Tripp Type 6: Hints
Guidelines:
- MUST be a declarative sentence — do NOT end with "?"
- State current condition, precondition, or desired state of the world
- Rely on situation knowledge and inference; easiest for listener to ignore
- Do NOT directly mention the action verb
- Distinct from `act_pragmatical_question_directive`, which is INTERROGATIVE
Examples:
- "pull the handle to open the drawer" → "the drawer is still closed"
- "use the switch to turn on the light bulb" → "the room could use some light"
- "grasp and lift the red block" → "the red block ought to be off the table by now"
""",
}
