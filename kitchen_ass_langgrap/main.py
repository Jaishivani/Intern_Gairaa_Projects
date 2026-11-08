from app.graph import graph


def main() -> None:
    print("Kitchen Assistant")
    print("Type your question, or press Ctrl+C to exit.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            print(
                "Assistant: Please enter a kitchen-related question."
            )
            continue

        try:
            result = graph.invoke(
                {
                    "question": question
                }
            )

            print(
                f"Assistant: {result['answer']}"
            )

        except (RuntimeError, ValueError) as error:
            print(
                f"Assistant: Error: {error}"
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nGoodbye.")