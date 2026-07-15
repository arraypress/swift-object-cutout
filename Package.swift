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
