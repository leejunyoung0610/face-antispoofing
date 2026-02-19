from tabulate import tabulate

def main():
    table1 = [
        ["Baseline", "94.75%", "N/A"],
        ["Texture CNN", "99.88%", "100.00%"],
        ["Frequency CNN", "96.81%", "N/A"],
        ["Multi-Expert", "99.00%", "N/A"],
    ]
    table2 = [
        ["Baseline", "66.67%", "97.75%", "91.25%", "21.75%", "17.50%"],
        ["Texture CNN", "89.38%", "N/A", "N/A", "22.87%", "N/A"],
        ["Frequency CNN", "86.04%", "N/A", "N/A", "41.88%", "N/A"],
        ["Multi-Expert", "87.29%", "N/A", "N/A", "N/A", "N/A"],
    ]
    table3 = [
        ["Baseline", "58.75%", "97.75%", "91.25%", "21.75%", "17.50%", "N/A", "N/A", "N/A"],
        ["Texture CNN", "98.75%", "98.50%", "98.54%", "1.38%", "1.25%", "97.50%", "98.25%", "97.88%"],
        ["Frequency CNN", "83.75%", "97.50%", "95.21%", "9.38%", "10.00%", "N/A", "N/A", "N/A"],
        ["2-Expert Static", "97.50%", "99.00%", "98.75%", "1.75%", "0.00%", "N/A", "N/A", "N/A"],
        ["Adaptive (τ=0.90)", "97.50%", "99.00%", "98.75%", "1.75%", "1.25%", "N/A", "N/A", "N/A"],
    ]
    table4 = [
        ["Texture CNN", "98.75%", "97.50%", "100.00%"],
        ["Frequency CNN", "82.50%", "97.08%", "98.12%"],
        ["Multi-Expert", "93.75%", "99.58%", "100.00%"],
    ]
    table5 = [
        ["Baseline", "66.67%", "83.33%", "+16.66%"],
        ["Texture CNN", "89.38%", "97.29%", "+7.91%"],
        ["Frequency CNN", "86.04%", "92.08%", "+6.04%"],
        ["Multi-Expert", "87.29%", "97.08%", "+9.79%"],
    ]

    print("=== Table 1: MSU-MFSD (Internal Dataset) ===")
    print(tabulate(table1, headers=["", "Frame-level", "Video-level"], tablefmt="github"))
    print("\n=== Table 2: Cross-dataset Generalization (No Target Training) ===")
    print(
        tabulate(
            table2,
            headers=["", "Live", "Spoof", "Overall", "HTER", "EER"],
            tablefmt="github",
        )
    )
    print("\n=== Table 3: Domain Adaptation (Target Fine-tuning) ===")
    print(
        tabulate(
            table3,
            headers=[
                "",
                "Live",
                "Spoof",
                "Overall",
                "HTER",
                "EER",
                "Video Live",
                "Video Spoof",
                "Video Overall",
            ],
            tablefmt="github",
        )
    )
    print("\n=== Table 4: Attack Type Analysis (Replay-Attack, Fine-tuned) ===")
    print(
        tabulate(
            table4,
            headers=["", "Live", "Print", "Display"],
            tablefmt="github",
        )
    )
    print("\n=== Table 5: Improvement Summary ===")
    print(
        tabulate(
            table5,
            headers=["", "Before Fine-tuning", "After Fine-tuning", "Improvement"],
            tablefmt="github",
        )
    )
    print(
        "\nKey Findings:\n"
        "- Texture CNN shows best generalization (-10% vs -28%)\n"
        "- Domain adaptation effective for all models\n"
        "- Video-level performance confirms no frame memorization\n"
        "- EER 0.00% achieved with 2-Expert system"
    )


if __name__ == "__main__":
    main()
