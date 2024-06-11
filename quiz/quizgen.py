import re

from quiz.quizcontent import quiz_content


def reformat_quiz_questions(content):
    # Regular expression to match each question block
    question_regex = r"# Question (\d+)(.*?)\n(1\. .*?)(\n2\. .*?)(\n3\. .*?)(\n4\. .*?)(\n\* Correct answer: \d+)"
    matches = re.findall(question_regex, content, re.DOTALL)

    formatted_questions = []

    i = 1

    for match in matches:
        question_number, question_title, ans1, ans2, ans3, ans4, correct_ans = match
        correct_ans_number = int(correct_ans.strip().split()[-1])

        question_number = i

        # Create a list of answers and reorder so the correct one is first
        answers = [ans1, ans2, ans3, ans4]
        correct_answer = answers.pop(correct_ans_number - 1)  # adjust index and remove correct answer
        answers.insert(0, correct_answer)  # insert correct answer at beginning

        # Build the formatted question string
        formatted_question = f"{question_number}. {question_title.strip()}\n\n" \
                             f"A. {answers[0][3:]}\n" \
                             f"B. {answers[1][3:]}\n" \
                             f"C. {answers[2][3:]}\n" \
                             f"D. {answers[3][3:]}\n\n"
        formatted_questions.append(formatted_question)

        i = i+1

    return "\n".join(formatted_questions)

# Example use
formatted_content = reformat_quiz_questions(quiz_content)
print(formatted_content)
