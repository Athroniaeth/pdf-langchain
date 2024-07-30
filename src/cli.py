from typer import Typer

cli = Typer(no_args_is_help=True)


@cli.command()
def deploy():
    ...


if __name__ == "__main__":
    cli()
