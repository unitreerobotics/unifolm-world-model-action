import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


class RichLogger:
    def __init__(self, level: str = "INFO"):
        # Initialize the console for rich output
        self.console = Console()

        # Define log levels with corresponding priority
        self.levels = {
            "DEBUG": 0,  # Lowest level, all logs are displayed
            "INFO": 1,  # Standard level, displays Info and higher
            "SUCCESS": 2,  # Displays success and higher priority logs
            "WARNING": 3,  # Displays warnings and errors
            "ERROR": 4,  # Highest level, only errors are shown
        }

        # Set default log level, use INFO if the level is invalid
        self.level = self.levels.get(level.upper(), 1)

    def _log(self, level: str, message: str, style: str, emoji=None):
        # Check if the current log level allows this message to be printed
        if self.levels[level] < self.levels["INFO"]:
            return

        # Format the timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Create a styled message
        text = Text(f"[{timestamp}] [{level}] {message}", style=style)

        # Print the message to the console
        self.console.print(text)

    def _log(self, level: str, message: str, style: str, emoji: str = None):
        # Check if the current log level allows this message to be printed
        if self.levels[level] < self.levels["INFO"]:
            return

        # Format the timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # If emoji is provided, prepend it to the message
        if emoji:
            message = f"{emoji} {message}"

        # Create a styled message
        text = Text(f"[{timestamp}] [{level}] {message}", style=style)

        # Print the message to the console
        self.console.print(text)

    # Basic log methods
    def info(self, message: str, emoji: str | None = None):
        # If the level is INFO or higher, print info log
        if self.levels["INFO"] >= self.level:
            self._log("INFO", message, "bold cyan", emoji)

    def warning(self, message: str, emoji: str = "⚠️"):
        # If the level is WARNING or higher, print warning log
        if self.levels["WARNING"] >= self.level:
            self._log("WARNING", message, "bold yellow", emoji)

    def error(self, message: str, emoji: str = "❌"):
        # If the level is ERROR or higher, print error log
        if self.levels["ERROR"] >= self.level:
            self._log("ERROR", message, "bold red", emoji)

    def success(self, message: str, emoji: str = "🚀"):
        # If the level is SUCCESS or higher, print success log
        if self.levels["SUCCESS"] >= self.level:
            self._log("SUCCESS", message, "bold green", emoji)

    def debug(self, message: str, emoji: str = "🔍"):
        # If the level is DEBUG or higher, print debug log
        if self.levels["DEBUG"] >= self.level:
            self._log("DEBUG", message, "dim", emoji)

    # ========== Extended Features ==========
    # Display a message with an emoji
    def emoji(self, message: str, emoji: str = "🚀"):
        self.console.print(f"{emoji} {message}", style="bold magenta")

    # Show a loading animation for a certain period
    def loading(self, message: str, seconds: float = 2.0):
        # Display a loading message with a spinner animation
        with self.console.status(f"[bold blue]{message}...", spinner="dots"):
            time.sleep(seconds)

    # Show a progress bar for small tasks
    def progress(self, task_description: str, total: int = 100, speed: float = 0.02):
        # Create and display a progress bar with time elapsed
        with Progress(
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            # Add a task to the progress bar
            task = progress.add_task(f"[cyan]{task_description}", total=total)
            while not progress.finished:
                progress.update(task, advance=1)
                time.sleep(speed)


# ========== Singleton Logger Instance ==========
_logger = RichLogger()


# ========== Function-style API ==========
def log_info(message: str, emoji: str | None = None):
    _logger.info(message=message, emoji=emoji)


def log_success(message: str, emoji: str = "🚀"):
    _logger.success(message=message, emoji=emoji)


def log_warning(message: str, emoji: str = "⚠️"):
    _logger.warning(message=message, emoji=emoji)


def log_error(message: str, emoji: str = "❌"):
    _logger.error(message=message, emoji=emoji)


def log_debug(message: str, emoji: str = "🔍"):
    _logger.debug(message=message, emoji=emoji)


def log_emoji(message: str, emoji: str = "🚀"):
    _logger.emoji(message, emoji)


def log_loading(message: str, seconds: float = 2.0):
    _logger.loading(message, seconds)


def log_progress(task_description: str, total: int = 100, speed: float = 0.02):
    _logger.progress(task_description, total, speed)


if __name__ == "__main__":
    # Example usage:
    # Initialize logger instance
    logger = RichLogger(level="INFO")  # Set initial log level to INFO

    # Log at different levels
    logger.info("System initialization complete.")
    logger.success("Robot started successfully!")
    logger.warning("Warning: Joint temperature high!")
    logger.error("Error: Failed to connect to robot")
    logger.debug("Debug: Initializing motor controllers")

    # Display an emoji message
    logger.emoji("This is a fun message with an emoji!", emoji="🔥")

    # Display loading animation for 3 seconds
    logger.loading("Loading motor control data...", seconds=3)

    # Show progress bar for a task with 100 steps
    logger.progress("Processing task", total=100, speed=0.05)

    # You can also use different log levels with a higher level than INFO, like ERROR:
    logger = RichLogger(level="ERROR")

    # Only error and higher priority logs will be shown (INFO, SUCCESS, WARNING will be hidden)
    logger.info("This won't be displayed because the level is set to ERROR")
    logger.error("This error will be displayed!")
