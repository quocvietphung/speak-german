"use client";

import React, { useMemo } from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Heading,
  Wrap,
  WrapItem,
  Badge,
  Spinner,
  Card,
  Separator,
  Button,
  Icon,
  Skeleton,
  SkeletonText,
  VisuallyHidden,
  useToken,
} from "@chakra-ui/react";
import { useTheme } from "next-themes";
import { MdPlayCircle } from "react-icons/md";
import {
  CircularProgressbarWithChildren,
  buildStyles,
} from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface ScoreFeedbackProps {
  loading: boolean;
  score: number | null; // 0 - 100
  mistakes: string[];
  tip: string;
  teacherFeedback: string;
  transcript: string;
}

export default function ScoreFeedbackCard({
  loading,
  score,
  mistakes,
  tip,
  teacherFeedback,
  transcript,
}: ScoreFeedbackProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const { palette, gradeLabel } = useMemo(() => {
    if (score == null) return { palette: "gray" as const, gradeLabel: "—" };
    if (score >= 90) return { palette: "green" as const, gradeLabel: "Excellent" };
    if (score >= 80) return { palette: "teal" as const, gradeLabel: "Great" };
    if (score >= 70) return { palette: "blue" as const, gradeLabel: "Good" };
    if (score >= 60) return { palette: "orange" as const, gradeLabel: "Fair" };
    return { palette: "red" as const, gradeLabel: "Needs practice" };
  }, [score]);

  const [p500, p400, gray200, whiteAlpha300] = useToken("colors", [
    `${palette}.500`,
    `${palette}.400`,
    "gray.200",
    "whiteAlpha.300",
  ]);

  const pathColor = isDark ? p400 : p500;
  const trailColor = isDark ? whiteAlpha300 : gray200;

  const handleSpeak = () => {
    if ("speechSynthesis" in window && transcript) {
      const u = new SpeechSynthesisUtterance(transcript);
      u.lang = /[äöüÄÖÜß]/.test(transcript) ? "de-DE" : "en-US";
      u.rate = 1;
      u.pitch = 1;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
  };

  return (
    <Card.Root
      p={6}
      rounded="2xl"
      shadow="xl"
      borderWidth="1px"
      bgGradient="linear(to-br, white, gray.50)"
      _dark={{ bgGradient: "linear(to-br, gray.800, gray.900)", borderColor: "whiteAlpha.200" }}
    >
      <Card.Header mb={2}>
        <HStack justify="space-between" align="start" w="full">
          <VStack align="start" gap={1}>
            <Heading size="md">📊 Score & Feedback</Heading>
            <Text color="fg.muted" textStyle="sm">
              Automatic pronunciation scoring & improvement tips
            </Text>
          </VStack>
          <Badge colorPalette={palette} variant="subtle" rounded="full" px={3} py={1}>
            {gradeLabel}
          </Badge>
        </HStack>
      </Card.Header>

      <Card.Body>
        {loading && (
          <VStack w="full" gap={6} align="stretch">
            <HStack justify="center" gap={4}>
              <Spinner size="lg" />
              <Text>Analyzing your speech…</Text>
            </HStack>
            <HStack gap={6}>
              <Skeleton rounded="full" boxSize="148px" />
              <VStack flex="1" align="start" gap={3}>
                <SkeletonText noOfLines={2} w="70%" />
                <Skeleton h="10" w="full" rounded="md" />
                <SkeletonText noOfLines={3} w="90%" />
              </VStack>
            </HStack>
          </VStack>
        )}

        {!loading && score !== null && (
          <VStack align="stretch" gap={6}>
            <HStack gap={6} align="center">
              <Box w={{ base: "136px", md: "156px" }} h={{ base: "136px", md: "156px" }}>
                <CircularProgressbarWithChildren
                  value={score}
                  maxValue={100}
                  strokeWidth={12}
                  styles={buildStyles({
                    pathColor,
                    trailColor,
                    textColor: "currentColor",
                    pathTransitionDuration: 0.5,
                    strokeLinecap: "round",
                  })}
                >
                  <VStack gap={0} lineHeight="1" textAlign="center">
                    <Text fontSize={{ base: "2xl", md: "3xl" }} fontWeight="bold">
                      {score}%
                    </Text>
                    <Text color="fg.muted" textStyle="2xs">
                      {gradeLabel}
                    </Text>
                  </VStack>
                </CircularProgressbarWithChildren>
              </Box>
              <VisuallyHidden aria-live="polite">{`Score ${score} percent`}</VisuallyHidden>

              <VStack align="start" gap={2} flex="1" minW={0}>
                <Text color="fg.muted" textStyle="sm">
                  Summary
                </Text>
                <Text>
                  Your pronunciation score: <b>{score}%</b>.{" "}
                  {gradeLabel === "Needs practice"
                    ? "Review the mistakes and practice slowly in chunks."
                    : "Keep up the rhythm and clarity!"}
                </Text>
                <HStack gap={3} flexWrap="wrap">
                  <Badge colorPalette={mistakes.length ? "red" : "green"} variant="surface" rounded="md">
                    {mistakes.length} mistake{mistakes.length !== 1 ? "s" : ""}
                  </Badge>
                  <Badge colorPalette="purple" variant="surface" rounded="md">
                    Tip ready
                  </Badge>
                </HStack>
              </VStack>
            </HStack>

            <Separator />

            <Box>
              <Text fontWeight="medium" mb={2}>
                Your sentence (AI detected)
              </Text>
              {transcript ? (
                <VStack align="start" gap={3}>
                  <HStack gap={3} wrap="wrap">
                    <Button onClick={handleSpeak} colorPalette="teal" variant="solid" rounded="lg">
                      <Icon as={MdPlayCircle} />
                      <Text ml={2}>Play transcript</Text>
                    </Button>
                    <Badge variant="subtle" colorPalette="cyan" rounded="full" px={3}>
                      Auto-detected
                    </Badge>
                  </HStack>
                  <Text as="q" fontStyle="italic" color="fg.muted">
                    {transcript}
                  </Text>
                </VStack>
              ) : (
                <Text color="fg.muted">No transcript yet. Record first.</Text>
              )}
            </Box>

            <Separator />

            <Box>
              <Text fontWeight="medium">Mistake Words</Text>
              <Wrap mt={2} gap={2}>
                {mistakes.length > 0 ? (
                  mistakes.map((w, i) => (
                    <WrapItem key={`${w}-${i}`}>
                      <Badge colorPalette="red" variant="outline" rounded="md" px={2} py={1}>
                        {w}
                      </Badge>
                    </WrapItem>
                  ))
                ) : (
                  <Text color="fg.muted">No mistakes 🎉</Text>
                )}
              </Wrap>
            </Box>

            <Separator />

            <Box>
              <Text fontWeight="medium" mb={1}>
                Tip
              </Text>
              <Text>{tip || "Keep a steady pace and stress the key syllables."}</Text>
            </Box>

            {teacherFeedback && (
              <>
                <Separator />
                <Box>
                  <Text fontWeight="medium" mb={1}>
                    👩‍🏫 Teacher Feedback
                  </Text>
                  <Text whiteSpace="pre-wrap">{teacherFeedback}</Text>
                </Box>
              </>
            )}
          </VStack>
        )}

        {!loading && score === null && (
          <VStack align="center" gap={3} py={4}>
            <Text color="fg.muted">No score yet. Record and analyze.</Text>
          </VStack>
        )}
      </Card.Body>
    </Card.Root>
  );
}