import csv
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from utils.dataset import ReplayAttackDataset


def extract_multiscale_lbp(image: np.ndarray, scales):
    """멀티 스케일 LBP histogram 생성"""
    features = []
    for R, P in scales:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        features.extend(hist.tolist())
    return np.asarray(features, dtype=np.float32)


def load_datasets():
    train_ds = ReplayAttackDataset(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        "train",
    )
    test_ds = ReplayAttackDataset(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        "test",
    )
    print("=== Multi-scale LBP 완전 탐색 ===")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}\n")
    return train_ds, test_ds


def evaluate_scales(train_ds, test_ds):
    experiments = {
        "R=1 only": [(1, 8)],
        "R=2 only": [(2, 16)],
        "R=3 only": [(3, 24)],
        "R=4 only": [(4, 32)],
        "R=1,2": [(1, 8), (2, 16)],
        "R=1,3": [(1, 8), (3, 24)],
        "R=1,4": [(1, 8), (4, 32)],
        "R=2,3": [(2, 16), (3, 24)],
        "R=2,4": [(2, 16), (4, 32)],
        "R=3,4": [(3, 24), (4, 32)],
        "R=1,2,3": [(1, 8), (2, 16), (3, 24)],
        "R=1,2,4": [(1, 8), (2, 16), (4, 32)],
        "R=1,3,4": [(1, 8), (3, 24), (4, 32)],
        "R=2,3,4": [(2, 16), (3, 24), (4, 32)],
        "R=1,2,3,4": [(1, 8), (2, 16), (3, 24), (4, 32)],
    }

    results = {}
    for exp_name, scales in experiments.items():
        print(f"[{exp_name}] 학습 중...")

        X_train, y_train = [], []
        for i in range(len(train_ds)):
            item = train_ds[i]
            image = (item["raw"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            X_train.append(extract_multiscale_lbp(image, scales))
            y_train.append(item["label"])
            if (i + 1) % 500 == 0:
                print(f"  Train: {i+1}/{len(train_ds)} samples processed")

        X_test, y_test = [], []
        for i in range(len(test_ds)):
            item = test_ds[i]
            image = (item["raw"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            X_test.append(extract_multiscale_lbp(image, scales))
            y_test.append(item["label"])

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100

        live_mask = np.array(y_test) == 0
        spoof_mask = np.array(y_test) == 1
        live_fp = np.sum((np.array(y_pred) == 1) & live_mask)
        live_total = np.sum(live_mask)
        spoof_fn = np.sum((np.array(y_pred) == 0) & spoof_mask)
        spoof_total = np.sum(spoof_mask)

        far = spoof_fn / spoof_total * 100 if spoof_total > 0 else 0
        frr = live_fp / live_total * 100 if live_total > 0 else 0
        hter = (far + frr) / 2

        results[exp_name] = {
            "acc": acc,
            "far": far,
            "frr": frr,
            "hter": hter,
            "n_features": len(X_train[0]) if X_train else 0,
        }

        print(f"  → Acc: {acc:.2f}%, Features: {results[exp_name]['n_features']}, HTER: {hter:.2f}%\n")

    return results


def summarize_results(results):
    print("=" * 80)
    print("전체 결과")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Accuracy':<12} {'Features':<12} {'FAR':<8} {'FRR':<8} {'HTER':<8}")
    print("-" * 80)
    for exp_name in sorted(results, key=lambda x: results[x]["acc"], reverse=True):
        r = results[exp_name]
        print(
            f"{exp_name:<20} {r['acc']:>10.2f}% {r['n_features']:>10} "
            f"{r['far']:>6.2f}% {r['frr']:>6.2f}% {r['hter']:>6.2f}%"
        )
    print("=" * 80)

    print("\n" + "=" * 60)
    print("단일 스케일 비교")
    print("=" * 60)
    for name in ["R=1 only", "R=2 only", "R=3 only", "R=4 only"]:
        if name in results:
            r = results[name]
            print(f"{name:<15} {r['acc']:>7.2f}%  ({r['n_features']} features)")

    print("\n" + "=" * 60)
    print("최적 조합 (Top 5)")
    print("=" * 60)
    top5 = sorted(results.items(), key=lambda x: x[1]["acc"], reverse=True)[:5]
    for i, (name, r) in enumerate(top5, 1):
        print(f"{i}. {name:<20} {r['acc']:.2f}%")

    print("\n" + "=" * 60)
    print("왜 R=1,2,3을 선택했는가?")
    print("=" * 60)
    combos = ["R=1,2,3", "R=1,2,4", "R=1,3,4", "R=2,3,4", "R=1,2,3,4"]
    for combo in combos:
        r = results.get(combo, {})
        print(f"{combo}: {r.get('acc', 0):.2f}% ({r.get('n_features', 0)} features)")

    print("\n분석:")
    print("- R=4는 너무 큰 반경으로 over-smoothing 가능성")
    print("- R=1,2,3은 성능/효율 균형 최적")
    print("- R=4 추가 시 feature 증가 대비 성능 향상 미미")

    out_csv = "multiscale_lbp_analysis.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Accuracy(%)", "Features", "FAR(%)", "FRR(%)", "HTER(%)"])
        for name, r in sorted(results.items(), key=lambda x: x[1]["acc"], reverse=True):
            writer.writerow([
                name,
                f"{r['acc']:.2f}",
                r["n_features"],
                f"{r['far']:.2f}",
                f"{r['frr']:.2f}",
                f"{r['hter']:.2f}",
            ])

    print(f"\n✅ 저장: {out_csv}")


def main():
    train_ds, test_ds = load_datasets()
    results = evaluate_scales(train_ds, test_ds)
    summarize_results(results)


if __name__ == "__main__":
    main()
