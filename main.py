"""AdvancEye 2.0 - CLI Entry Point."""
import argparse
import sys
from pathlib import Path

from config.settings import AppSettings, CameraSettings, get_settings
from services.attendance_service import AttendanceService
from services.recognition_service import RecognitionService
from services.registration_service import RegistrationService
from storage.json_repository import JSONRepository
from ui.recognition_view import RecognitionView
from ui.registration_view import RegistrationView
from utils.logger import get_logger
from utils.video_stream import VideoStreamReader

logger = get_logger("AdvancEyeCLI")


def handle_register(args, settings: AppSettings) -> None:
    """Handle student enrollment CLI command."""
    logger.info(f"Starting enrollment for roll number: {args.roll} ({args.name})")
    repo = JSONRepository(settings.storage)
    reg_service = RegistrationService(repo, settings=settings)

    cam_settings = CameraSettings(stream_source=args.source or settings.camera.stream_source)
    stream = VideoStreamReader(cam_settings)

    view = RegistrationView(reg_service, stream, settings=settings)
    success = view.run_registration(roll_no=args.roll, name=args.name)

    if success:
        print(f"\n[SUCCESS] Student {args.name} ({args.roll}) successfully registered!")
    else:
        print(f"\n[INFO] Enrollment session terminated.")


def handle_run(args, settings: AppSettings) -> None:
    """Handle live recognition and attendance monitoring CLI command."""
    logger.info("Starting live attendance monitoring...")
    repo = JSONRepository(settings.storage)
    att_service = AttendanceService(repo, attendance_settings=settings.attendance, camera_settings=settings.camera)
    rec_service = RecognitionService(repo, att_service, settings=settings)

    cam_settings = CameraSettings(stream_source=args.source or settings.camera.stream_source)
    stream = VideoStreamReader(cam_settings)

    view = RecognitionView(rec_service, stream, settings=settings)
    view.run_live_recognition()


def handle_list(args, settings: AppSettings) -> None:
    """Handle student profile listing CLI command."""
    repo = JSONRepository(settings.storage)
    profiles = repo.list_all_profiles()
    print(f"\n=== Registered Student Profiles ({len(profiles)}) ===")
    print(f"{'Roll No':<20} | {'Name':<25} | {'Cells':<8} | {'Enrolled At'}")
    print("-" * 75)
    for p in profiles:
        print(f"{p.get('roll_no', 'N/A'):<20} | {p.get('name', 'N/A'):<25} | {p.get('total_cells_captured', 0):<8} | {p.get('enrolled_at', 'N/A')}")
    print("-" * 75)


def handle_attendance(args, settings: AppSettings) -> None:
    """Handle viewing attendance logs CLI command."""
    repo = JSONRepository(settings.storage)
    records = repo.get_attendance_records(date_str=args.date)
    print(f"\n=== Attendance Records for {args.date or 'Today'} ({len(records)}) ===")
    print(f"{'Roll No':<20} | {'Name':<25} | {'Confidence':<10} | {'Timestamp'}")
    print("-" * 75)
    for r in records:
        print(f"{r.get('roll_no', 'N/A'):<20} | {r.get('name', 'N/A'):<25} | {r.get('confidence_similarity', 0.0):<10.3f} | {r.get('timestamp', 'N/A')}")
    print("-" * 75)


def main() -> None:
    """CLI Argument Parser and Command Dispatcher."""
    parser = argparse.ArgumentParser(description="AdvancEye 2.0 Biometric Attendance System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: register
    reg_parser = subparsers.add_parser("register", help="Enroll a new student profile")
    reg_parser.add_argument("--roll", required=True, help="Student unique roll number (e.g. 2026_CS_042)")
    reg_parser.add_argument("--name", required=True, help="Student full name")
    reg_parser.add_argument("--source", default=None, help="Video stream source / webcam index")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Start real-time attendance monitoring")
    run_parser.add_argument("--source", default=None, help="Video stream source / webcam index / RTSP URL")

    # Command: list
    subparsers.add_parser("list", help="List all enrolled student profiles")

    # Command: attendance
    att_parser = subparsers.add_parser("attendance", help="View attendance records")
    att_parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD format (default: today)")

    args = parser.parse_args()
    settings = get_settings()

    if args.command == "register":
        handle_register(args, settings)
    elif args.command == "run":
        handle_run(args, settings)
    elif args.command == "list":
        handle_list(args, settings)
    elif args.command == "attendance":
        handle_attendance(args, settings)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
