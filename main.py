from agent import ExerciseAgent
from functions import fetch_video_id, fetch_transcript, clean_transcript, words_dictionary, classify_cefr


def main():
    print("Welcome to your teaching assistant")

    #preparing the linguistic analysis

    url = input("Enter YouTube URL of the video you would like to use in your class: ")
    try:
        video_id = fetch_video_id(url)
        transcript = fetch_transcript(video_id)
    except Exception as e:
        print(f"\nSorry, I couldn't get the transcript for that video: {e}")
        return

    final_transcript = clean_transcript(transcript)
    word_by_pos = words_dictionary(final_transcript)
    CEFR = classify_cefr(final_transcript) #this returns a dictionary
    CEFR_LEVEL = CEFR["level"]


    print("\n\n")
    print(f"""I analyzed your video!
    The video contains the following words organized by their lexical categories
    {word_by_pos}
    And its CEFR level is {CEFR_LEVEL}""")
    print("\n\n")
    print("Now, it's time to prepare some exercises :)")

    agent = ExerciseAgent()
    result = agent.run(transcript=final_transcript, vocab=word_by_pos, level=CEFR_LEVEL)
    print("Well, nice chat. Goodbye!")


if __name__ == '__main__':
    main()