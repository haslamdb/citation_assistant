#!/usr/bin/env python3
"""
User Management CLI
Create and manage users for Citation Assistant
"""

import sys
import getpass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auth import user_manager


def create_user():
    """Interactive user creation"""
    print("\n=== Create New User ===\n")

    username = input("Username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return

    # Check if user exists
    if user_manager.get_user(username):
        print(f"Error: User '{username}' already exists")
        return

    email = input("Email (optional): ").strip() or None
    full_name = input("Full name (optional): ").strip() or None

    # Get password
    while True:
        password = getpass.getpass("Password: ")
        if len(password) < 8:
            print("Error: Password must be at least 8 characters")
            continue

        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("Error: Passwords do not match")
            continue

        break

    try:
        user = user_manager.create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name
        )
        print(f"\n✓ User '{username}' created successfully!")

    except Exception as e:
        print(f"\nError creating user: {e}")


def list_users():
    """List all users"""
    users = user_manager.list_users()

    if not users:
        print("\nNo users found.")
        return

    print("\n=== Users ===\n")
    for username in users:
        user = user_manager.get_user(username)
        status = "disabled" if user.disabled else "active"
        email = f" ({user.email})" if user.email else ""
        print(f"  - {username}{email} [{status}]")

    print()


def delete_user():
    """Delete a user"""
    print("\n=== Delete User ===\n")

    username = input("Username to delete: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return

    if not user_manager.get_user(username):
        print(f"Error: User '{username}' not found")
        return

    confirm = input(f"Are you sure you want to delete '{username}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled")
        return

    try:
        user_manager.delete_user(username)
        print(f"\n✓ User '{username}' deleted successfully!")

    except Exception as e:
        print(f"\nError deleting user: {e}")


def change_password():
    """Change user password"""
    print("\n=== Change Password ===\n")

    username = input("Username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return

    if not user_manager.get_user(username):
        print(f"Error: User '{username}' not found")
        return

    # Get new password
    while True:
        password = getpass.getpass("New password: ")
        if len(password) < 8:
            print("Error: Password must be at least 8 characters")
            continue

        password_confirm = getpass.getpass("Confirm new password: ")
        if password != password_confirm:
            print("Error: Passwords do not match")
            continue

        break

    try:
        user_manager.update_password(username, password)
        print(f"\n✓ Password for '{username}' changed successfully!")

    except Exception as e:
        print(f"\nError changing password: {e}")


def main():
    """Main menu"""
    while True:
        print("\n" + "=" * 50)
        print("Citation Assistant - User Management")
        print("=" * 50)
        print("\n1. Create user")
        print("2. List users")
        print("3. Delete user")
        print("4. Change password")
        print("5. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            create_user()
        elif choice == '2':
            list_users()
        elif choice == '3':
            delete_user()
        elif choice == '4':
            change_password()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid option")


if __name__ == "__main__":
    # Quick command-line creation
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        if len(sys.argv) >= 4:
            username = sys.argv[2]
            password = sys.argv[3]
            email = sys.argv[4] if len(sys.argv) > 4 else None

            try:
                user_manager.create_user(username, password, email=email)
                print(f"✓ User '{username}' created successfully!")
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            print("Usage: python manage_users.py create <username> <password> [email]")
            sys.exit(1)
    else:
        main()
