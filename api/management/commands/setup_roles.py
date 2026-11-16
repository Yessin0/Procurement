"""Create the API role groups and optionally put a user in one.

    python manage.py setup_roles
    python manage.py setup_roles --user alice --role analyst
Admin = superuser/staff (or the 'admin' group); viewers only see /api/metrics/*.
"""
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management.base import BaseCommand, CommandError

from api.permissions import ROLES


class Command(BaseCommand):
    help = "Create the admin/analyst/viewer groups; optionally assign a user to one."

    def add_arguments(self, parser):
        parser.add_argument("--user")
        parser.add_argument("--role", choices=ROLES)

    def handle(self, *args, **options):
        for name in ROLES:
            Group.objects.get_or_create(name=name)
        self.stdout.write(f"Groups ready: {', '.join(ROLES)}")
        if options["user"] or options["role"]:
            if not (options["user"] and options["role"]):
                raise CommandError("Use --user and --role together.")
            try:
                user = get_user_model().objects.get(username=options["user"])
            except get_user_model().DoesNotExist:
                raise CommandError(f"No user named {options['user']!r}.")
            user.groups.add(Group.objects.get(name=options["role"]))
            self.stdout.write(self.style.SUCCESS(f"{user.get_username()} is now {options['role']}."))