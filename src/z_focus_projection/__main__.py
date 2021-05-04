"""Click CLI app"""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Z Focus Projection."""


if __name__ == "__main__":
    main(prog_name="z-focus-projection")
