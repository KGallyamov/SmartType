from datasets import load_dataset
from typing import Iterator, Literal


class CodeSearchNetDataset:
    """Wrapper for CodeSearchNet dataset that filters by programming language."""

    SUPPORTED_LANGUAGES = {
        'go': 'Go programming language',
        'java': 'Java programming language',
        'javascript': 'Javascript programming language',
        'php': 'PHP programming language',
        'python': 'Python programming language',
        'ruby': 'Ruby programming language'
    }

    def __init__(self,
                 language: Literal['go', 'java', 'javascript', 'php', 'python', 'ruby'],
                 split: str = "train",
                 return_field: str = "whole_func_string"):
        """
        Args:
            language: Programming language to filter ('go', 'java', 'javascript', 'php', 'python', 'ruby')
            split: Dataset split (default: "train")
            return_field: Field to return - options: "whole_func_string", "func_code_string",
                         "func_documentation_string", or any other field name
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of {list(self.SUPPORTED_LANGUAGES.keys())}")

        self.language = language
        self.language_full = self.SUPPORTED_LANGUAGES[language]
        self.split = split
        self.return_field = return_field

        # Load the full dataset
        self.dataset = load_dataset(
            "code-search-net/code_search_net",
            language,
            split=split,
            streaming=False,
            trust_remote_code=True
        )

        # Filter by language
        # self.dataset = full_dataset.filter(
        #     lambda x: x['language'] == self.language_full
        # )

    def __getitem__(self, idx: int | slice):
        """Get the specified field at the given index."""
        if isinstance(idx, int):
            return self.dataset[idx][self.return_field]
        elif isinstance(idx, slice):
            sliced = CodeSearchNetDataset.__new__(CodeSearchNetDataset)
            sliced.return_field = self.return_field
            sliced.dataset = self.dataset.select(range(*idx.indices(len(self.dataset))))
            return sliced

    def __len__(self) -> int:
        """Return the number of items in the filtered dataset."""
        return len(self.dataset)

    def __iter__(self) -> Iterator[str]:
        """Iterate through the dataset, yielding the specified field."""
        for item in self.dataset:
            yield item[self.return_field]


class SyntheticCppDataset:
    """Wrapper for the ReySajju742/synthetic-cpp dataset."""

    def __init__(self,
                 split: str = "train",
                 return_field: Literal["prompt", "response"] = "response"):
        """
        Args:
            split: Dataset split to use (default: "train")
            return_field: Field to return - options: "prompt", "response"
        """
        self.split = split
        self.return_field = return_field

        # Load the specified dataset from Hugging Face
        self.dataset = load_dataset(
            "ReySajju742/synthetic-cpp",
            split=split,
            streaming=False,
        )

    def __getitem__(self, index: int) -> str:
        return self.dataset[index][self.return_field]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)

    def __iter__(self) -> Iterator[str]:
        """Iterate through the dataset, yielding the specified field."""
        for item in self.dataset:
            yield item[self.return_field]


# Example usage:
if __name__ == "__main__":
    # Python functions
    python_dataset = CodeSearchNetDataset(language="python", split="train")
    print(f"Python dataset size: {len(python_dataset)}")
    print("Python function:")
    print(python_dataset[0][:200])
    print("\n" + "=" * 50 + "\n")

    # Java functions
    java_dataset = CodeSearchNetDataset(language="java", split="train")
    print(f"Java dataset size: {len(java_dataset)}")
    print("Java function:")
    print(java_dataset[0][:200])
    print("\n" + "=" * 50 + "\n")

    # Get only docstrings
    docs_dataset = CodeSearchNetDataset(
        language="python",
        split="train",
        return_field="func_documentation_string"
    )
    print("Python docstring:")
    print(docs_dataset[0])
