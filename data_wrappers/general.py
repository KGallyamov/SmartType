from datasets import load_dataset
from typing import Iterator, Optional


class CasualDataset:
    """Wrapper for streaming datasets that provides index-based text access."""

    def __init__(self, dataset_name: str, name: Optional[str] = None,
                 split: str = "train", text_field: str = "text"):
        """
        Args:
            dataset_name: Dataset to load (e.g., "HuggingFaceFW/fineweb", "fancyzhx/ag_news")
            name: Dataset configuration (e.g., "CC-MAIN-2024-10", "sample-10BT")
            split: Dataset split (default: "train")
            text_field: Field containing text data (default: "text")
        """
        self.dataset_name = dataset_name
        self.name = name
        self.split = split
        self.text_field = text_field

        self.dataset = load_dataset(
            dataset_name,
            name=name,
            split=split,
            streaming=True
        )

        self._cache = {}

    def __getitem__(self, index: int) -> str:
        """Get text at the specified index."""
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")

        if index < 0:
            raise IndexError("Negative indexing is not supported for streaming datasets")

        if index in self._cache:
            return self._cache[index]

        max_cached = max(self._cache.keys()) if self._cache else -1

        if index <= max_cached:
            return self._cache[index]

        # Stream to reach the desired index
        iterator = iter(self.dataset.skip(max_cached + 1))

        for i in range(max_cached + 1, index + 1):
            try:
                item = next(iterator)
                text = item[self.text_field]
                self._cache[i] = text
            except StopIteration:
                raise IndexError(f"Index {index} is out of bounds for the dataset")

        return self._cache[index]

    def __iter__(self) -> Iterator[str]:
        """Iterate through the dataset, yielding text strings."""
        for i, item in enumerate(self.dataset):
            text = item[self.text_field]
            self._cache[i] = text
            yield text

    def clear_cache(self):
        """Clear the internal cache to free memory."""
        self._cache.clear()


# Example usage:
if __name__ == "__main__":
    # FineWeb dataset
    fineweb = CasualDataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train"
    )
    print(f"FineWeb[0]: {fineweb[0][:100]}...")

    # AG News dataset
    ag_news = CasualDataset(
        "fancyzhx/ag_news",
        split="train"
    )
    print(f"AG News[0]: {ag_news[0][:100]}...")
    print(f"AG News[5]: {ag_news[5][:100]}...")