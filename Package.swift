// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ObjectCutout",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        // On-device "tap any object → crisp cutout" via SAM 2 (Apache).
        // Click a point; get a transparent-background cutout of that object.
        .library(name: "ObjectCutout", targets: ["ObjectCutout"]),
    ],
    targets: [
        .target(name: "ObjectCutout"),
        .testTarget(name: "ObjectCutoutTests", dependencies: ["ObjectCutout"]),
    ]
)

// Swift 7's import rule, today: a file must import every module whose members
// it uses, rather than leaning on another file's imports.
for target in package.targets where target.type != .system && target.type != .binary && target.type != .plugin {
    target.swiftSettings = (target.swiftSettings ?? []) + [.enableUpcomingFeature("MemberImportVisibility")]
}
